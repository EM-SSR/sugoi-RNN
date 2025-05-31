import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch import nn, distributions as torchd
import torch.nn.functional as F
import contextlib

import networks
import tools

from ffjord.odenvp import ODENVP                      # flow-based エンコーダ／デコーダ
from dnc.dnc_sam_ctm import DNC                        # CTM 組み込み DNC
from dnc.controller import BaseController   # 標準 Controller

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

def _cfg_get(obj, key, default=None):
    """dict / SimpleNamespace / その他 に共通の安全な取得"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# ── FlowEncoder / FlowDecoder ─────────────────────────
class FlowEncoder(nn.Module):
    """
    ODENVP ベースの可逆エンコーダ。
    - 推論時のみ GPU に小さなチャンクを載せ、ピークメモリを大幅削減
    - 出力ベクトルが self.outdim より短い場合は 0-padding し
      後段 (DNC など) の input_size と必ず一致させる
    """
    def __init__(self, img_shape_CHW, cfg):
        super().__init__()
        C, H, W = img_shape_CHW

        # ------ Flow 本体 -------------------------------------------------
        self.flow = ODENVP(
            input_size=(1, C, H, W),
            n_blocks=_cfg_get(cfg, "n_blocks", 2),
            intermediate_dims=tuple(_cfg_get(cfg, "hidden", (32,))),
            nonlinearity=_cfg_get(cfg, "nonlinearity", "softplus"),
            squash_input=True,
            cnf_kwargs=dict(T=0.5, solver="rk4", atol=1e-3, rtol=1e-3),
        )

        # ------ 出力次元を事前に測定 ------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W, device=next(self.flow.parameters()).device)
            self.outdim = self.flow(dummy).shape[-1]

        self._use_amp = True

    # ------------------------------------------------------------------
    def forward(self, obs_dict):
        # obs_dict["image"]: (B,L,H,W,C) or (B,H,W,C)
        x = obs_dict["image"]
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,H,W,C)
        # HWC→CHW
        if x.shape[-1] in (1, 3):
            B, L, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # →(B,L,C,H,W)
        else:
            B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)  # (N, C, H, W)
        device = next(self.flow.parameters()).device
        x = x.to(device, non_blocking=True).float()
        autocast = torch.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext
        with autocast(device_type="cuda", enabled=self._use_amp):
            z = self.flow(x)  # 一括で流す
        if z.shape[-1] < self.outdim:
            z = F.pad(z, (0, self.outdim - z.shape[-1]), "constant", 0.0)
        return z.view(B, L, self.outdim)  # (B, L, D_fixed)
    
class FlowDecoder(nn.Module):
    """
    逆方向 ODENVP で画像を再構成し、ContDist でラップして
    .mode() が呼び出せる確率分布を返す。
    """
    def __init__(self, feat_dim, img_shape_CHW, flow_obj):
        super().__init__()
        self.flow = flow_obj.flow         # 共有パラメータを再利用
        self.CHW  = img_shape_CHW         # (C,H,W)
        self._use_amp = True

    def forward(self, feat):
        # feat: (B, L, D)
        B, L, D = feat.shape
        x = feat.reshape(B * L, D)
        device = next(self.flow.parameters()).device
        x = x.to(device, non_blocking=True).float()
        autocast = torch.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext
        with autocast(device_type="cuda", enabled=self._use_amp):
            rec = self.flow(x, reverse=True)  # (N, C, H, W)
        C, H, W = self.CHW
        x_rec = rec.view(B, L, C, H, W).permute(0, 1, 3, 4, 2)  # (B, L, H, W, C)
        dist = tools.ContDist(
            torch.distributions.Independent(
                torch.distributions.Normal(x_rec, 1e-2), 3
            )
        )
        return {"image": dist}
    
# ── DNCDynamics ─────────────────────────────────
class DNCDynamics(nn.Module):
    """
    RSSM 互換の DNC ラッパ  
    ・`img_step` を追加（Dreamer の imagination 用）  
    ・`obs_step` で初回の `prev_action is None` を安全に処理
    """
    def __init__(self, in_dim, act_dim, cfg):
        super().__init__()
        self.act_dim = act_dim
        self._feat_dim = in_dim
        self._dist_cls = torchd.Normal

        self.dnc = DNC(
            controller_class=BaseController,
            input_size=in_dim + act_dim,
            output_size=in_dim,
            nn_output_size=cfg.dyn_deter,
            mem_slot=cfg.dnc_mem_slot,
            mem_size=cfg.dnc_mem_size,
            read_heads=cfg.dnc_read_heads,
            batch_size=cfg.batch_size,
            n_transformer_layers=cfg.dnc_transformer_layers,
            use_cuda=cfg.device.startswith("cuda"),
        )

    # --- core ----------------------------------------------------
    def observe(self, embed, actions, is_first):
        seq = torch.cat([embed, actions], -1)                 # (B,L,D+A)
        outs, _ = self._run_seq(seq)
        post  = {"stoch": outs, "deter": torch.zeros_like(outs)}
        return post, {k: v.clone() for k, v in post.items()}

    def obs_step(self, prev_state, prev_action, embed_t, is_first_t):
        # embed_t : (B,D) または (B,1,D) が来るので単一ステップに統一
        if embed_t.dim() == 3 and embed_t.shape[1] == 1:
            embed_t = embed_t.squeeze(1)          # → (B,D)

        if prev_action is None:                   # 初回ステップ用ダミー
            prev_action = torch.zeros(
                embed_t.shape[0], self.act_dim, device=embed_t.device
            )

        inp = torch.cat([embed_t, prev_action], -1).unsqueeze(0)  # (1,B,D+A)
        stoch = self.dnc(inp)[0].squeeze(0)
        state = {"stoch": stoch, "deter": torch.zeros_like(stoch)}
        return state, state

    def img_step(self, prev_state, prev_action):
        """観測なしの 1 ステップ予測"""
        B = prev_state["stoch"].shape[0]
        if prev_action is None:
            prev_action = torch.zeros(B, self.act_dim, device=prev_state["stoch"].device)
        dummy = torch.zeros(B, self._feat_dim, device=prev_state["stoch"].device)
        inp = torch.cat([dummy, prev_action], -1).unsqueeze(0)    # (1,B,D+A)
        stoch = self.dnc(inp)[0].squeeze(0)
        return {"stoch": stoch, "deter": torch.zeros_like(stoch)}

    def imagine_with_action(self, actions, init):
        B, L, _ = actions.shape
        dummy = torch.zeros(B, L, self._feat_dim, device=actions.device)
        outs, _ = self._run_seq(torch.cat([dummy, actions], -1))
        return {"stoch": outs, "deter": torch.zeros_like(outs)}

    # --- utilities ---------------------------------------------
    def _run_seq(self, seq):
        out, sup = self.dnc(seq.transpose(0, 1))
        return out.transpose(0, 1), sup

    def get_feat(self, state):
        return state["stoch"]

    def get_dist(self, state):
        mean = state["stoch"]
        std = torch.ones_like(mean) * 0.1
        return torchd.Independent(self._dist_cls(mean, std), 1)

    def kl_loss(self, *args, **kwargs):
        zeros = torch.zeros_like(args[0]["stoch"][..., 0])
        return zeros, zeros, zeros, zeros
    
# ── WorldModel 置き換え版 ───────────────────────────────
class WorldModel(nn.Module):
    """
    Flow + CTM-DNC ベースの WorldModel
    * 画像 shape を HWC/CHW 自動判定
    * self._use_amp を保持（_train 内で参照される）
    * それ以外のロジックはこれまでと同一
    """
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._config, self._step = config, step
        self._use_amp = bool(getattr(config, "precision", 32) == 16)

        # ---- 画像 shape 取得 & 自動変換 -----------------
        img_shape = tuple(obs_space.spaces["image"].shape)    # (H,W,C) or (C,H,W)
        if img_shape[0] in (1, 3):                            # CHW
            C, H, W = img_shape
        else:                                                 # HWC
            H, W, C = img_shape
        img_CHW = (C, H, W)

        # ---- Flow Encoder ---------------------------------
        ode_cfg = getattr(config, "odenvp", {})
        ode_cfg = SimpleNamespace(**{
            "n_blocks": 2, "hidden": (16, 16), "nonlinearity": "softplus", **ode_cfg
        })
        self.encoder = FlowEncoder(img_CHW, ode_cfg)
        self.embed_size = self.encoder.outdim

        # ---- DNC Dynamics ---------------------------------
        dnc_cfg = SimpleNamespace(
            dnc_mem_slot=getattr(config, "dnc_mem_slot", 16),
            dnc_mem_size=getattr(config, "dnc_mem_size", 16),
            dnc_read_heads=getattr(config, "dnc_read_heads", 4),
            dnc_transformer_layers=getattr(config, "dnc_transformer_layers", 3),
            dyn_deter=getattr(config, "units", 200),
            batch_size=getattr(config, "batch_size", 8),
            device=getattr(config, "device", "cuda"),
        )
        self.dynamics = DNCDynamics(
            self.embed_size,
            getattr(config, "num_actions", act_space),
            dnc_cfg,
        )
        
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = FlowDecoder(self.embed_size, img_CHW, self.encoder)
        feat_dim   = self.embed_size
        r_cfg      = getattr(config, "reward_head", None)
        c_cfg      = getattr(config, "cont_head",   None)
        units      = getattr(config, "units", 256)
        act_fn     = getattr(config, "act",   "SiLU")
        norm_type  = getattr(config, "norm",  "none")
        device     = getattr(config, "device", "cuda")

        self.heads["reward"] = networks.MLP(
            feat_dim, (),                    # in_dim, out_dim(=())
            _cfg_get(r_cfg, "layers", 2),
            units, act_fn, norm_type,
            dist=_cfg_get(r_cfg, "dist", "normal"),
            outscale=_cfg_get(r_cfg, "outscale", 1.0),
            device=device, name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_dim, (),                    # binary continuation
            _cfg_get(c_cfg, "layers", 2),
            units, act_fn, norm_type,
            dist="binary",
            outscale=_cfg_get(c_cfg, "outscale", 1.0),
            device=device, name="Cont",
        )

        # ---------- Optimizer -------------------------
        self._model_opt = tools.Optimizer(
            "model", self.parameters(),
            getattr(config, "model_lr", 6e-4),
            getattr(config, "opt_eps", 1e-8),
            getattr(config, "grad_clip", 100.0),
            getattr(config, "weight_decay", 0.0),
            opt=getattr(config, "opt", "adam"),
            use_amp=True,
        )
        self._scales = dict(
            reward=_cfg_get(r_cfg, "loss_scale", 1.0),
            cont=_cfg_get(c_cfg, "loss_scale", 1.0),
        )

        
    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        """
        * 画像はまだ GPU に乗せず、uint8 → float16 変換のみ行う
          （実際に GPU に転送するのは FlowEncoder 内で
           chunk 単位になった瞬間）
        * その他の小さな項目だけ先に GPU へ。
        """
        obs_proc = {}
        for k, v in obs.items():
            if k == "image":
                # uint8 CPU → float16 CPU, 値域 [0,1]
                obs_proc["image"] = (
                    torch.from_numpy(v).to(dtype=torch.float16) / 255.0
                )
            else:
                obs_proc[k] = torch.tensor(
                    v, device=self._config.device, dtype=torch.float32
                )

        if "discount" in obs_proc:
            obs_proc["discount"] *= self._config.discount
            obs_proc["discount"] = obs_proc["discount"].unsqueeze(-1)

        assert "is_first" in obs_proc
        assert "is_terminal" in obs_proc
        obs_proc["cont"] = (1.0 - obs_proc["is_terminal"]).unsqueeze(-1)
        return obs_proc
    
    def video_pred(
        self,
        data,
        *,                 # 可視化に使う範囲
        vis_batch=4,       # 何シーケンス分を可視化するか
        obs_frames=5,      # 観測フレーム数
        pred_frames=6,     # 想像フレーム数
    ):
        data = self.preprocess(data)

        # --- ① 必要な範囲だけ残す -----------------------------------
        keep_T = obs_frames + pred_frames
        data = {
            k: (v[:vis_batch, :keep_T].contiguous()
                if torch.is_tensor(v) and v.ndim >= 2 else v)
            for k, v in data.items()
        }

        # --- ② 軽量エンコード ---------------------------------------
        embed = self.encoder.forward(
            {"image": data["image"]},
        )

        # --- ③ 以降は従来どおり（slice だけ合わせる） ---------------
        states, _ = self.dynamics.observe(
            embed[:, :obs_frames],
            data["action"][:, :obs_frames],
            data["is_first"][:, :obs_frames],
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()
        init  = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(
            data["action"][:, obs_frames:], init
        )
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()

        real = data["image"].to(recon.device)
        model = torch.cat([recon, openl], 1)
        error = (model - real + 1.0) / 2.0
        return torch.cat([real, model, error], 2)
    
class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model

        # 元々は config.dyn_stoch + config.dyn_deter などで計算していましたが、
        # 実際の潜在表現次元と合わせるため、world_model.embed_size を使います。
        feat_size = world_model.embed_size

        # Actor ネットワーク
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        # Value ネットワーク
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        # ------------------

        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
