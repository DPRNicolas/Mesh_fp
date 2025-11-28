import numpy as np
import Converter.Internal as I
import Converter.PyTree as C
import Post.PyTree as P
import Transform as T
import os
from scipy.interpolate import griddata

def comp_Sutherland(propref, Ts, Cs, T):
    '''Dynamical viscosity / thermal conductivity from sutherland law'''
    return propref*np.sqrt(T/Ts)*((1.+Cs/Ts)/(1.+Cs/T))
def compute_prim_bf(bf, gam=1.4, r_gas=287.15):
    
    bfp = np.zeros(bf.shape, dtype=np.float64)
    #primitives (rho u T)
    bft = np.zeros(bf.shape, dtype=np.float64)

    bfp[0] = bf[0]
    bfp[1:4] = bf[1:4]/bf[0]
    
    bft = np.copy(bfp)
    
    bfp[-1] = (gam-1.0) * (bf[-1]-0.5*(np.sum(bf[1:4]**2, axis=0))/bf[0])

    
    bft[-1] = bfp[-1] / (r_gas * bft[0])

    # T =  (gam-1.0)*(bf[-1]-0.5*(np.sum(bf[1:4]**2, axis=0))/bf[0])/(r_gas*bf[0])
    # print('bfp[-1]-T=', norm(bfp[-1]-T))
    # print('bft[-1]-T=', norm(bft[-1]-T))
    
    return bfp, bft


file_dir = "/net/jabba/home1/nd612731/Documents/mesh/mesh_flat_plate"
file = "fixedpoint_dim.cgns"
out_dir = "/net/jabba/home1/nd612731/Documents/mesh/mesh_flat_plate"
out_file = "FP_Ma_4.5_Re_m_3.4e6_Ni_200_Nj_150_TM.cgns"
out_file = "FP_Ma_4.5_Re_m_3.4e6_Ni_1000_Nj_150_TM.cgns"


names = ["Density", "MomentumX", "MomentumY", "MomentumZ", "EnergyStagnationDensity"]


###
# Physical setup
dphys = dict()
dphys['gam']      = 1.4       # Gamma constant
dphys['Ts']       = 273.15    # Sutherland ref temp
dphys['cs']       = 110.4     # Sutherland constant (temp
dphys['musuth']   = 1.715e-5  # Sutherland ref dynamic viscosity
dphys['rgaz']     = 287.056     # Air ideal gas constant
dphys['Prandtl']  = 0.72      # Prandtl number


path_file = os.path.join(file_dir, file)
print('input:', path_file) 
t_base = C.convertFile2PyTree(path_file)

path_file_out = os.path.join(out_dir, out_file)
print('output:', path_file_out) 
t_out = C.convertFile2PyTree(path_file_out)
t_out2 = I.copyTree(t_out)


I.printTree(t_out)
print('')

adim_in = False

if adim_in:
    ref = I.getNodeFromName(t_out, name='ReferenceState')
    Lref=I.getVal(I.getNodeFromName(ref, 'Length_ref'))
    print('Lref = ', Lref)
    Lref = 1 / I.getVal(I.getNodeFromName(ref, "Reynolds_unit"))


    rho_ref = I.getVal(I.getNodeFromName(ref, 'Density_ref'))
    mom_ref = I.getVal(I.getNodeFromName(ref, 'V_ref'))*rho_ref
    rhoE_ref = I.getVal(I.getNodeFromName(ref, 'EnergyStagnation_ref'))

else: 
    Lref=1.0
    rho_ref = 1.0
    mom_ref = 1.0
    rhoE_ref = 1.0

#
stateref= {"Density": rho_ref, "MomentumX": mom_ref, "MomentumY":mom_ref, "MomentumZ":mom_ref, "EnergyStagnationDensity": rhoE_ref}
print("stateref: ", stateref)





t_base = C.node2Center(t_base)
t_out = C.node2Center(t_out)

grid_x_in_2D = I.getValue(I.getNodeFromName(t_base, "CoordinateX"))*Lref
grid_y_in_2D = I.getValue(I.getNodeFromName(t_base, "CoordinateY"))*Lref
values_in = I.getValue(I.getNodeFromName(t_base, "Density"))*rho_ref
grid_x_in = np.reshape(grid_x_in_2D, (-1, 1), order="F")
grid_y_in = np.reshape(grid_y_in_2D, (-1, 1), order="F")
values_in = np.squeeze(np.reshape(values_in, (-1, 1), order="F"))


#Wall coordinates is x,y base
X_wall_node_in = grid_x_in_2D[:,0]
Y_wall_node_in = grid_y_in_2D[:,0]
X_wall_in = (X_wall_node_in[1:]+X_wall_node_in[:-1])/2
Y_wall_in = (Y_wall_node_in[1:]+Y_wall_node_in[:-1])/2
Rn_in = np.abs(X_wall_in[0])
print(f'{Rn_in = }')





points_in = np.transpose(np.squeeze((grid_x_in, grid_y_in)))
print(np.shape(points_in))
print(np.shape(values_in))

# output file
grid_x_out = I.getValue(I.getNodeFromName(t_out, "CoordinateX"))
grid_y_out = I.getValue(I.getNodeFromName(t_out, "CoordinateY"))
print("shape grid out = ", np.shape(grid_x_out))
values_out = I.getValue(I.getNodeFromName(t_out, "Density"))

points_out = (grid_x_out, grid_y_out)
#Wall coordinates is x,y base
X_wall_node = grid_x_out[:,0]
Y_wall_node = grid_y_out[:,0]
X_wall = (X_wall_node[1:]+X_wall_node[:-1])/2
Y_wall = (Y_wall_node[1:]+Y_wall_node[:-1])/2
Rn = np.abs(X_wall[0])
print(f'{Rn = }')

shape = np.shape(I.getValue(I.getNodeFromName(t_out2, "Density")))
nshape = [5]
[nshape.append(i) for i in shape]
sol = np.zeros(nshape)
sol_nearest = np.zeros(nshape)
sol[0] = I.getValue(I.getNodeFromName(t_out2, "Density"))
sol[1] = I.getValue(I.getNodeFromName(t_out2, "MomentumX"))
sol[2] = I.getValue(I.getNodeFromName(t_out2, "MomentumY"))
sol[4] = I.getValue(I.getNodeFromName(t_out2, "EnergyStagnationDensity"))


if 1:
    im_in, jm_in = np.shape(grid_x_in_2D)
    bf_in = np.zeros((im_in, jm_in, 5))

    for i in range(5):
        bf_in[:,:,i] = I.getValue(I.getNodeFromName(t_base, names[i]))*stateref[names[i]]
    bf_in = np.moveaxis(bf_in, (0,1,2), (1,2,0))
    bfp_in, bft_in = compute_prim_bf(bf_in, dphys['gam'],dphys['rgaz'])

    mu = comp_Sutherland(dphys['musuth'], dphys['Ts'], dphys['cs'], bft_in[4,:,:])
    print('rho inf = {}, MumX inf = {}, MumY inf = {}, Estag inf = {}'.format(bf_in[0,0,-1], bf_in[1,0,-1], bf_in[2,0,-1], bf_in[4,0,-1]))
    print('rho inf = {}, Vx inf = {}, Vy inf = {}, T inf = {}'.format(bft_in[0,0,-1], bft_in[1,0,-1], bft_in[2,0,-1], bft_in[4,0,-1]))

    ds_j_wall_in = ((grid_x_in_2D[:, 1]-X_wall_node_in)**2+(grid_y_in_2D[:, 1]-Y_wall_node_in)**2)**(0.5)
    V_in = (bft_in[1,:,:]**2+bft_in[2,:,:]**2)**0.5


    tau_wall = mu[:,0]*(V_in[:,1]-0)/ds_j_wall_in
    U_tau = (tau_wall/bft_in[0,:,0])**(0.5)
    y_plus_1st_cell = U_tau*ds_j_wall_in*bft_in[0,:,0]/mu[:,0]



    euclid_len = lambda x,y: np.sqrt(x**2+ y**2)
    dist2center = euclid_len(X_wall,Y_wall) #Calculate the distance of the wall to the origini 0=(0,0)
    insphere = np.isclose(dist2center, Rn, rtol=1e-04, atol=1e-09, equal_nan=False) #find points on the nose sphere
    # i_cone_junction = np.squeeze(np.argwhere(insphere == True)[-1])
    # print(f'{i_cone_junction = }')
    print('\nFor input mesh')
    print(f'At stagnation point \t: y+ = {y_plus_1st_cell[0]}, \tU_tau = {U_tau[0]}, \ttau_wall = {tau_wall[0]}')
    # print(f'At cone junction point \t: y+ = {y_plus_1st_cell[i_cone_junction]}, \tU_tau = {U_tau[i_cone_junction]}, \ttau_wall = {tau_wall[i_cone_junction]}, {i_cone_junction = }')
    print(f'At end \t\t\t: y+ = {y_plus_1st_cell[-1]}, \tU_tau = {U_tau[-1]}, \ttau_wall = {tau_wall[-1]}')


# interpolation
for i in range(5):
    print('interp '+names[i])
    values_in = I.getValue(I.getNodeFromName(t_base, names[i]))*stateref[names[i]]
    values_in = np.squeeze(np.reshape(values_in, (-1, 1), order="F"))
    sol[i] = griddata(points_in, values_in, points_out, method="nearest", fill_value= np.nan)
    sol_nearest[i] = griddata(points_in, values_in, points_out, method="nearest")

#If nans (because out of bounds => nearest interp)
sol[np.isnan(sol)] = sol_nearest[np.isnan(sol)]

I.setValue(I.getNodeFromName(t_out2, "Density"), sol[0])
I.setValue(I.getNodeFromName(t_out2, "MomentumX"), sol[1])
I.setValue(I.getNodeFromName(t_out2, "MomentumY"), sol[2])
I.setValue(I.getNodeFromName(t_out2, "EnergyStagnationDensity"), sol[4])






#Pseudo y+ for laminar flow to use for the mesh

im_out, jm_out = np.shape(grid_x_out)
bf_out = np.zeros((im_out, jm_out, 5))
for i in range(5):
    bf_out[:,:,i] = sol[i]
bf_out = np.moveaxis(bf_out, (0,1,2), (1,2,0))
bfp_out, bft_out = compute_prim_bf(bf_out, dphys['gam'],dphys['rgaz'])

mu = comp_Sutherland(dphys['musuth'], dphys['Ts'], dphys['cs'], bft_out[4,:,:])
print('rho inf = {}, MumX inf = {}, MumY inf = {}, Estag inf = {}'.format(bf_out[0,0,-1], bf_out[1,0,-1], bf_out[2,0,-1], bf_out[4,0,-1]))
print('rho inf = {}, Vx inf = {}, Vy inf = {}, T inf = {}'.format(bft_out[0,0,-1], bft_out[1,0,-1], bft_out[2,0,-1], bft_out[4,0,-1]))

ds_j_wall = ((grid_x_out[:, 1]-X_wall_node)**2+(grid_y_out[:, 1]-Y_wall_node)**2)**(0.5)
V_out = (bft_out[1,:,:]**2+bft_out[2,:,:]**2)**0.5

tau_wall = mu[:,0]*(V_out[:,1]-0)/ds_j_wall
U_tau = (tau_wall/bft_out[0,:,0])**(0.5)
y_plus_1st_cell = U_tau*ds_j_wall*bft_out[0,:,0]/mu[:,0]

euclid_len = lambda x,y: np.sqrt(x**2+ y**2)
dist2center = euclid_len(X_wall,Y_wall) #Calculate the distance of the wall to the origini 0=(0,0)
insphere = np.isclose(dist2center, Rn, rtol=1e-04, atol=1e-09, equal_nan=False) #find points on the nose sphere
i_cone_junction = np.squeeze(np.argwhere(insphere == True)[-1])
i_cone_junction = int(len(y_plus_1st_cell)/2)
print(f'{i_cone_junction = }')
print('\nFor output mesh')
print(f'At stagnation point \t: y+ = {y_plus_1st_cell[0]}, \tU_tau = {U_tau[0]}, \ttau_wall = {tau_wall[0]}')
print(f'At midle \t: y+ = {y_plus_1st_cell[i_cone_junction]}, \tU_tau = {U_tau[i_cone_junction]}, \ttau_wall = {tau_wall[i_cone_junction]}, {i_cone_junction = }')
print(f'At end \t\t\t: y+ = {y_plus_1st_cell[-1]}, \tU_tau = {U_tau[-1]}, \ttau_wall = {tau_wall[-1]}')


def find_n_sol_and_interp(quantity, eta, j_max_shock_normal, n=1, value2find=[0], ignore = 0):
    j_array = []
    i_array = []
    eta_array = []
    i_error = []
    if len(value2find) == np.shape(quantity)[0]:
        for i,column in enumerate(quantity):
            try:
                to_append = np.where(np.nan_to_num(np.diff(np.sign(column[ignore:j_max_shock_normal[i]]-value2find[i]))))[0]+ignore
                for k, el in enumerate(to_append):
                    if k <n: #take only the n first solution even though there shouldn't be any other
                        j_array.append(el) #detects sign change to  isolate index before '0' and ignores nans but counts there index
                        i_array.append(i)
                        eta_array.append((value2find[i] - quantity[i, el])*((eta[i, el+1] - eta[i, el])/(quantity[i, el+1]-quantity[i, el]))+eta[i, el])
            except:
                i_error.append(i)
    elif np.shape(value2find)[0] == 1 and (type(j_max_shock_normal) is int or type(j_max_shock_normal) is type(None)):
        for i,column in enumerate(quantity):
            if i == 900:
                pass
            try:
                to_append = np.where(np.nan_to_num(np.diff(np.sign(column[ignore:j_max_shock_normal]-value2find))))[0]+ignore
                for k, el in enumerate(to_append):
                    if k <n: #take only the n first solution even though there shouldn't be any other
                        j_array.append(el) #detects sign change to  isolate index before '0' and ignores nans but counts their index
                        i_array.append(i)
                        eta_array.append((value2find - quantity[i, el])*((eta[i, el+1] - eta[i, el])/(quantity[i, el+1]-quantity[i, el]))+eta[i, el])
            except:
                i_error.append(i)
    elif np.shape(j_max_shock_normal)[0] == 1:
        for i,column in enumerate(quantity):
            try:
                to_append = np.where(np.nan_to_num(np.diff(np.sign(column[ignore:j_max_shock_normal[i]]-value2find))))[0]+ignore
                for k, el in enumerate(to_append):
                    if k <n: #take only the n first solution even though there shouldn't be any other
                        j_array.append(el) #detects sign change to  isolate index before '0' and ignores nans but counts their index
                        i_array.append(i)
                        eta_array.append((value2find - quantity[i, el])*((eta[i, el+1] - eta[i, el])/(quantity[i, el+1]-quantity[i, el]))+eta[i, el])
            except:
                i_error.append(i)
    else:
        print('Error: Value 2 find dimension are not standard')

    print('Solution not found for:', i_error)
    j_array = np.array(j_array)
    i_array = np.array(i_array)
    eta_array = np.array(eta_array)
    print('solution shape: ', i_array.shape)

    return i_array, j_array, eta_array
i_delta_u99, j_delta_u99, eta_delta_u99 = find_n_sol_and_interp(V_out, grid_y_out, -1, n=1, value2find=[0.99*V_out[-1,-1]], ignore = 0)
name_out =  out_file.split(".cgns")[0] + "_interp_yplus_{:.2f}_{:.2f}_{:.2f}_ptsCL_{}.cgns".format(y_plus_1st_cell[0], y_plus_1st_cell[i_cone_junction], y_plus_1st_cell[-1], j_delta_u99[-1])
C.convertPyTree2File(t_out2, os.path.join(out_dir, name_out))