import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from cerebras.modelzoo.models.gnn.pyg_gnn.eval import evaluate

def train_model(cfg, model, loaders, data, split_idx, device):
    init = cfg["trainer"]["init"]
    model_dir = init["model_dir"]
    
    # Optimizer
    optconf = init["optimizer"]["AdamW"]
    optimizer = AdamW(
        model.parameters(),
        lr=optconf["learning_rate"],
        weight_decay=optconf["weight_decay"],
    )

    # AMP
    m = init["model"]
    use_amp = bool(m.get("to_float16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Config Options
    compute_eval_metrics = m.get("compute_eval_metrics", True)
    disable_log_softmax = m.get("disable_log_softmax", False)

    # Loop
    loop = init["loop"]
    max_steps = int(loop["max_steps"])
    steps_per_epoch = int(loop["steps_per_epoch"])
    eval_frequency = int(loop["eval_frequency"])
    grad_accum = int(loop.get("grad_accum_steps", 1))

    log_steps = cfg["trainer"]["init"]["logging"]["log_steps"]

    train_loader, val_loader = loaders

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch_size=batch.batch_size)
            logits = logits[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            if not disable_log_softmax:
                logits = F.log_softmax(logits, dim=-1)
                loss = F.nll_loss(logits, y)
            else:
                loss = F.cross_entropy(logits, y)

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % log_steps == 0:
            avg = running_loss / log_steps
            print(f"[step {step:04d}] loss={avg:.4f}")
            running_loss = 0.0

        if step % steps_per_epoch == 0:
            epoch += 1

        if compute_eval_metrics and (step % eval_frequency == 0 or step == max_steps):
            val_acc = evaluate(model, val_loader, device)
            print(f"[eval @ step {step}] val_acc={val_acc:.4f}")
            # Ensure model is back in train mode after eval
            model.train()

    # Save checkpoint after final evaluation (which happens at step == max_steps above)
    ckpt = os.path.join(model_dir, "last.pt")
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt)
    print(f"Saved: {ckpt}")
