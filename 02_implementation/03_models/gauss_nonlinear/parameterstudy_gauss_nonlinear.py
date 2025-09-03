"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import numpy as np
print(np.__version__)
import deepxde as dde
print(dde.backend)
import matplotlib.pyplot as plt
from scipy import io
import pandas as pd
import time
import mat73
import os
import csv


data = mat73.loadmat(r"/home/users/jbi/Bachelorarbeit/Codebase/Netze/MIONet/annular_P_v_IC_nl_data/annular_P_v_IC_vvar_nl_10k_4k_20x.mat")

p_train = data["P_train"].astype(np.float32)
p_test = data["P_test"].astype(np.float32)

ic_train = data["IC_train"].astype(np.float32)
ic_test = data["IC_test"].astype(np.float32)

v_train = data["v_train"].astype(np.float32)
v_test = data["v_test"].astype(np.float32)

branch_train = np.concatenate([ic_train, v_train, p_train], axis=1)
branch_test = np.concatenate([ic_test, v_test, p_test], axis=1)

u_bias_train= data["U_bias_train"].astype(np.float32)
u_bias_test= data["U_bias_test"].astype(np.float32)
xyz_t_bias_train = data["xyz_t_bias_train"].astype(np.float32)
xyz_t_bias_test = data["xyz_t_bias_test"].astype(np.float32)

u_rd_train= data["U_rd_train"].astype(np.float32)
u_rd_test= data["U_rd_test"].astype(np.float32)
xyz_t_rd_train = data["xyz_t_rd_train"].astype(np.float32)
xyz_t_rd_test = data["xyz_t_rd_test"].astype(np.float32)

u_v_train= data["U_v_train"].astype(np.float32)
u_v_test= data["U_v_test"].astype(np.float32)
xyz_t_v_train = data["xyz_t_v_train"].astype(np.float32)
xyz_t_v_test = data["xyz_t_v_test"].astype(np.float32)

xyz_t_train = np.concatenate((xyz_t_bias_train, xyz_t_rd_train, xyz_t_v_train), axis=0)
xyz_t_test = np.concatenate((xyz_t_bias_test, xyz_t_rd_test, xyz_t_v_test), axis=0)

u_train = np.concatenate((u_bias_train, u_rd_train, u_v_train), axis=1)
u_test = np.concatenate((u_bias_test, u_rd_test, u_v_test), axis=1)

print(xyz_t_train.shape,ic_train.shape,p_train.shape,v_train.shape,branch_train.shape,u_train.shape)
print(xyz_t_test.shape,ic_test.shape,p_test.shape,v_test.shape,branch_test.shape,u_test.shape)

def train_deeponet(config,model_id):
    # Modellverzeichnis
    model_dir = f"/home/users/jbi/Bachelorarbeit/Codebase/Netze/MIONet/annular_P_v_IC_nl_models/Parameterstudien_vvar/P_v_IC_nl_V2/{model_id}"
    os.makedirs(model_dir, exist_ok=True)

    n_train = config["n_train"]
    n_test = n_train//4

    y_train = u_train[:n_train]
    y_test = u_test[:n_test]

    x_train = (ic_train[:n_train], v_train[:n_train], p_train[:n_train], xyz_t_train)
    x_test = (ic_test[:n_test], v_test[:n_test], p_test[:n_test], xyz_t_test)

    data = dde.data.QuintupleCartesianProd(x_train, y_train, x_test, y_test)

    # Netzwerk
    x1,x2,x3,x4 = 400,20,20,4

    neurons = config["neurons"]
    layers = config["layers"]
    act_IC = config["activation IC"]
    act_branch = config["activation Branch"]
    act_trunk = config["activation Trunk"]
    merge_operation = config["merge_operation"]
    output_merge_operation = config["output_merge_operation"]
    decay = config["decay"]

    if merge_operation == "cat":
        b_neurons = int(neurons/2)
    else:
        b_neurons = neurons

    net = dde.nn.MIONetCartesianProd_3Branches(
                        [x1] + [neurons] * (layers-1) + [b_neurons], 
                        [x2] + [neurons] * (layers-1) + [b_neurons],
                        [x3] + [neurons] * (layers-1) + [b_neurons],
                        [x4] + [neurons] * layers, 
                        {"branch1":act_IC,"branch2":act_branch,"branch3":act_branch,"trunk":act_trunk}, 
                        "Glorot normal",
                        regularization=None,
                        merge_operation=merge_operation,
                        output_merge_operation=output_merge_operation
                    )

    start_time = time.time()

    model = dde.Model(data, net)

    if decay == "cosine":
        model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"],decay=("cosine", 100_000, 0))
    else:
        model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"])

    os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
    checker = dde.callbacks.ModelCheckpoint(
    f"{model_dir}/checkpoints/model", save_better_only=True, period=10000
    )

    losshistory, train_state = model.train(iterations=config["iterations"],display_every=1000,callbacks=[checker])

    end_time = time.time()
    
    # ✅ Speichere Modell
    directory = model.save(f"{model_dir}/model")

    # ✅ Loss Plot
    plt.figure()
    plt.semilogy(losshistory.steps, losshistory.loss_train, label="Train loss")
    if losshistory.loss_test:
        plt.semilogy(losshistory.steps, losshistory.loss_test, label="Test loss")
    if losshistory.metrics_test:
        plt.semilogy(losshistory.steps, losshistory.metrics_test, label="mean l2 relative error")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss history: {model_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/loss_plot.png")
    plt.close()

    # Optional: Speichere Verlustwerte als CSV
    np.savetxt(f"{model_dir}/loss_train.csv", np.array(losshistory.loss_train), delimiter=",")
    if losshistory.loss_test:
        np.savetxt(f"{model_dir}/loss_test.csv", np.array(losshistory.loss_test), delimiter=",")
    if losshistory.metrics_test:
        np.savetxt(f"{model_dir}/l2_relative_error.csv", np.array(losshistory.metrics_test), delimiter=",")
        
    result = {
                    "final_loss_train": losshistory.loss_train[-1][0],
                    "final_loss_test": losshistory.loss_test[-1][0],
                    "l2_relative_error": losshistory.metrics_test[-1][0],
                    "train_time_s": round(end_time - start_time, 2),
                    "train_time_h": round((end_time - start_time)/3600, 2)
                }

    return model,losshistory.loss_train[-1][0], result, directory


configs = []
id = 0


#n_train in (8000,800):
for neurons in (200,100,300):
    layers = 5 #for layers in (4,6):
    for act_IC in ("relu","swish"):
        for act_branch in ("relu","swish"):
                n_train = 8000
                act_trunk ="relu"# for act_trunk in ("relu","swish"):
                decay = "cosine"#for decay in ("None","cosine"):
                merge_operation = "mul" #for merge_operation in ("mul","add","cat"):
                output_merge_operation = "mul" #for merge_operation in ("mul","add","cat"):
                iterations = 100_000
                config = {
                    #"model_id": f"model_{model_id:04d}",
                    "n_train": n_train,
                    "iterations": iterations,
                    "neurons": neurons,
                    "layers": layers,
                    "activation IC": act_IC,
                    "activation Branch": act_branch,
                    "activation Trunk": act_trunk,
                    "merge_operation": merge_operation,
                    "output_merge_operation": output_merge_operation,
                    "decay": decay

                }
                configs.append(config)
                id += 1
                
print(configs[id-1])
print(id)


results = []

for i, config in enumerate(configs):
    model_id = f"model_{i:02d}"
    print(f"\n🔍 Testing config {i+1}/{len(configs)}: {config}")
    try:
        model,loss,result,dir = train_deeponet(config,model_id)
        print(f"✅ Done: Final training loss = {loss:.4e}")
        print(model,loss,result,dir)
        results.append({**config, **result, **{"directory": dir}})
    except Exception as e:
        print(f"⚠️ Error: {e}")
        results.append(config)
    
    print(results[0])
    keys = results[0].keys()
    with open('/home/users/jbi/Bachelorarbeit/Codebase/Netze/MIONet/annular_P_v_IC_nl_models/Parameterstudien_vvar/P_v_IC_nl_V2/train_results.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)