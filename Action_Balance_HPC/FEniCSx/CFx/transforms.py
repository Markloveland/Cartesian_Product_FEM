import numpy as np

#functions containing transforms from 2 subdomains 
# to 1 global domain that is the cartesian product of the 2 subdomains

def cartesian_product_coords(array1,array2):
    #gives cartesian product of 2 vectors
    #input is meant to handle np.arrays of coordinates
    #where the points are listed row by row

    #convention is (array1[0],array2[0] ; array1[0],array2[1] ; ....)
    n1,dim1=array1.shape
    n2,dim2=array2.shape
    out_arr=np.zeros((n1*n2,dim1+dim2))
    c=0
    for a in range(n1):
        for b in range(n2):
            out_arr[c,:]=np.append(array1[a,:],array2[b,:])
            c=c+1
    return out_arr

def cartesian_product_dofs(array1,array2):
    #gives cartesian product of 2 1d vectors
    #same as above but meant for vector not array input
    dim1=len(array1)
    dim2=len(array2)
    out_arr=np.zeros((dim1*dim2,2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=[array1[a],array2[b]]
            c=c+1
    return out_arr


def cartesian_form_to_kroneck_form(indeces, len_dim_2):
    #designed to take list of indexes in the cartesian form
    #translate to single form as kron product would
    #only for cartesian product between 2 spaces
    #expects matrix with n rows, 2 columns
    num_indeces=indeces.shape[0]
    out_arr=np.zeros(num_indeces)
    for n in range(num_indeces):
        out_arr[n] = indeces[n,0]*len_dim_2 + indeces[n,1]
    return out_arr


