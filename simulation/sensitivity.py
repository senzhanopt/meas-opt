import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D  # for 3D plots

temperature = pd.read_csv("data\\temperature_5minute.csv")["temp (C)"].to_numpy()

def sensitivity_voltage(soc, p_max, delta = 0.1):
    
    direc = "data\\voltage_" + str(p_max) +'.npy'
    data_voltage = np.load(direc)
    
    soc_up = soc + 0.1
    soc_down = soc - 0.1
    soc_up = min(soc_up, 1.0)
    soc_down = max(soc_down, 0.0)
        
    l = []
    for i in range(1,len(data_voltage)):
        if data_voltage[i,0] <= soc_up and data_voltage[i,0] >= soc_down and abs(data_voltage[i,1])>=1e-4:
            sens = (data_voltage[i,2]-data_voltage[i-1,2])/data_voltage[i,1]
            l.append(sens)
            
    return max(l)


def sensitivity_thermal(p_max):
     
    direc = "data\\temperature_" + str(p_max) +'.npy'
    data_temp = np.load(direc)
    
    power = data_temp[1:,1]
    power = power**2
    temp_delta = data_temp[1:,2] - data_temp[:-1,2]
    temp_ambient = np.tile(temperature,7)
    temp_delta_ambient = data_temp[:-1,2] - temp_ambient[:-1]
    
    X = np.concatenate((temp_delta_ambient.reshape(-1, 1), power.reshape(-1, 1)),axis=1)
    y = temp_delta
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    return model.coef_

def sensitivity_thermal_reg(p_max):
     
    direc = "data\\temperature_" + str(p_max) +'.npy'
    data_temp = np.load(direc)
    
    power = data_temp[1:,1]
    power = power**2
    temp_delta = data_temp[1:,2] - data_temp[:-1,2]
    temp_ambient = np.tile(temperature,7)
    temp_delta_ambient = data_temp[:-1,2] - temp_ambient[:-1]
    
    X = np.concatenate((temp_delta_ambient.reshape(-1, 1), power.reshape(-1, 1)),axis=1)
    y = temp_delta
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # ---------- Metrics ----------
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("\n--- Model Metrics ---")
    print(f"Coefficients [temp_delta_ambient, power²]: {model.coef_}")
    print(f"R² score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # ---------- Cross-validation ----------
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R² (mean ± std): {scores.mean():.4f} ± {scores.std():.4f}")

    '''
    # ---------- Visualization ----------
    # 1️⃣ Scatter plots of predictors vs target
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(temp_delta_ambient, y, alpha=0.7, color='blue')
    ax[0].set_xlabel("Temp Delta Ambient")
    ax[0].set_ylabel("Temp Delta")
    ax[0].set_title("Temp Delta vs Temp Delta Ambient")

    ax[1].scatter(power, y, alpha=0.7, color='red')
    ax[1].set_xlabel("Power²")
    ax[1].set_ylabel("Temp Delta")
    ax[1].set_title("Temp Delta vs Power²")
    plt.tight_layout()
    plt.show()

    # 2️⃣ Residuals vs predicted
    residuals = y - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Temp Delta")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.show()
    '''
    # 3️⃣ 3D scatter + regression plane
    fig = plt.figure(figsize=(7.5,4.5))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(temp_delta_ambient, power, y, c='purple', alpha=0.6)

    xx, yy = np.meshgrid(
        np.linspace(temp_delta_ambient.min(), temp_delta_ambient.max(), 20),
        np.linspace(power.min(), power.max(), 20)
    )
    zz = model.coef_[0] * xx + model.coef_[1] * yy
    ax3d.plot_surface(xx, yy, zz, color='orange', alpha=0.4)

    ax3d.set_xlabel("Temperature diff.\n to ambient (°C)")
    ax3d.set_zlabel("Temperature change (°C)")
    ax3d.set_ylabel("Power squared")
    #ax3d.zaxis.labelpad=-5.0
    ax3d.set_box_aspect(aspect=None, zoom=0.83)
    plt.savefig(f"..\\..\\paper\\tempdyn{p_max}.pdf", bbox_inches='tight')
    plt.show()

    return model.coef_

if __name__ == '__main__':
    sensitivity_thermal_reg(6.8)