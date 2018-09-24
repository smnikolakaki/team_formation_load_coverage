import numpy as np
import time
import random
import math
import pandas as pd
import os

from pulp import *
from cvxopt import matrix
from cvxopt import solvers
from cvxopt import sparse
from cvxopt import spmatrix
from numpy import array, eye, hstack, ones, vstack, zeros
from pylab import random as pyrandom
from numpy  import array
from scipy.sparse import csr_matrix
from collections import Counter

np.set_printoptions(threshold=np.nan)


def create_load_constraint(P,J,m,k,l,n,max_load):
    '''
    Ensures that the load of each person does not exceed the maximum Load.
    That is the sum of job participation of each person does not exceed L.
    Input: People,Jobs,Number of jobs, Number of people, Number of skills, PeoplexJobs,max load
    Output: A (each row corresponds to one constraint), b (each element is the RHS of a constraint)
    '''
    # max_load = max_load - 1
    
    A = zeros((k,n+1)) 
    # print A.shape
    b = zeros((k))
    
    idx = 0
    for i in range(0,k):
        A[i,idx:idx+m] = 1.0
        idx = idx+m
     
    A[:, n] = -ones(k)
    for i in range(0,k):
        b[i] = float(0)
    
#     print 'Constraints for load:'
#     print 'Al is:',A
#     print
#     print 'bl is:',b
#     print
    
    return A,b

def create_domain_constraint(P,J,m,k,l,n,minx,maxx,max_load):
    '''
    Writes constraints $x_{i,j}\in [0,1]$ in a matrix vector form acceptable by cvxopt
    Input: People,Jobs,Number of jobs, Number of people, Number of skills, PeoplexJobs,min possible value, max possible value
    Output: A (each row corresponds to one constraint), b (each element is the RHS of a constraint)
    '''
    A = zeros((2*(n+1),n+1)) 
    b = zeros((2*(n+1)))
    
    for i in range(0,(n+1)):
        A[i,i] = -1.0
    
    for i in range((n+1),2*(n+1)-1):
        A[i,i-(n+1)] = 1.0
    
    x_min = minx * ones(n+1)
    x_max = maxx * ones(n+1)
    
    b = hstack([-x_min,x_max]) 
    
    b[n] = -0
    b[2*n+1] = float(max_load)  
    
    return A,b


# cross products
def cross(a,b): 
    p = 0
    for (x,y) in zip(a,b):
        p+=(x*y)
    return p

# formalize the output from linear program solver
def output_pulp(arr): 
    new = np.zeros((len(arr),len(arr[0])))
    for i in range(len(new)):
        for j in range(len(new[0])):
            new[i][j] = value(arr[i][j])
    return new


def output_cvxopt(X,k,m):
    new = np.zeros((k,m))
    count = 0
    for i in range(0,k):
        for j in range(0,m):
            new[i][j] = X[count]
            count+=1
     
    tot = new.shape[0]*new.shape[1]
    zeros = tot - np.count_nonzero(new)
    
    return new,X[len(X)-1],zeros


def create_people_tasks(file_people,file_tasks):
    J = []; P = [];
    J = np.genfromtxt(file_tasks, delimiter=',')
    P = np.genfromtxt(file_people, delimiter=',')
    P = P.astype(int)
    J = J.astype(int)
    
    return J,P


def convert_data(J,P):
    JD = []; PD = []
    for idx,task in enumerate(J):
        new = []
        for idx2,skill in enumerate(task):
            if skill == 1:
                new.append(idx2+1)
        
        JD.append(new)
        
    for idx,task in enumerate(P):
        new = []
        for idx2,skill in enumerate(task):
            if skill == True:
                new.append(idx2+1)
                
        PD.append(new)
    
    return (JD,PD)


def expert_project_skills(P_orig, J_orig):
    """
    Creates a list with all skills encountered in experts.
    Input: the experts list of lists
    Output: a dictionary of the skills in experts
    """
    skills = []; exp_skills_existence = {}; proj_skills_existence = {}
    for pid,pskill in enumerate(P_orig):
        skills+=pskill
     
    skills = list(set(skills))
    
    for skill in skills:
        exp_skills_existence[skill] = 1
     
    skills = [];
    for jid,jskill in enumerate(J_orig):
        skills+=jskill
     
    skills = list(set(skills))
    
    for skill in skills:
        proj_skills_existence[skill] = 1
        
    return exp_skills_existence,proj_skills_existence


def create_cover_constraint(P,J,m,k,l,n):
    '''
    Ensures that the skill requirements of each job are covered.
    That is the sum of people skill participating per job skill is equal to 1.
    Input: People,Jobs,Number of jobs, Number of people, Number of skills, PeoplexJobs
    Output: A (each row corresponds to one constraint), b (each element is the RHS of a constraint)
    '''
    
    A = zeros((m*l,n+1)) 
    b = zeros((m*l)) 
    
    lrow = []
    lcol = []
    ldata = []
    
    print 'Creating lists for A matrix in cover constraint'     
    count = 0; row = 0; ocol = 0; r = 0
    for i,person_skills in enumerate(P):
        row = 0; col = ocol
        for j,skill in enumerate(person_skills):
            while count < m:
#                 if skill != 0.0:
#                     lrow.append(row+r)
#                     lcol.append(col)
#                     ldata.append(-skill)
                    
                A[row+r, col] = -skill
                r+=l
                col+=1
                count+=1
                
            count = 0; r = 0; row+=1; col = ocol
            
        ocol+=m
    
    print 'Creating b vector in cover constraint' 
    count = 0
    for i,job_skills in enumerate(J):
        for j,skill in enumerate(job_skills): 
            b[count] = -skill
            count+=1
            
    print 'Converting lists to np arrays in cover constraint'
#     lrow = np.array(lrow)
#     lcol = np.array(lcol)
#     ldata = np.array(ldata)
    
#     lendata = len(ldata)
#     lenrow = len(lrow)
#     lencol = len(lcol)
    print 'Creating the csr matrix in cover constraint'
#     A = csr_matrix((ldata, (lrow, lcol))) 
    print 'Finally returning A and b in cover constraint matrix'
    return A,b

def remove_nonoverlap_skils(J,P,exp_skills_existence,proj_skills_existence):
    """
        Removes from projects all skills that are not covered by any expert and vice versa
        Input: the project dictionary and the expert skills dictionary, the expert dictionary and the project skills dictionary
        Output: the updated projects, the updated experts
    """    
    
    # j: begins indexing from 0 but skills in dictionary begin from 1
    for i in range(0,J.shape[0]):
        for j in range(0,J.shape[1]):
            if (J[i,j] == 1) and (j+1 not in exp_skills_existence):
                J[i,j] = 0
    
    remove_list = []
    for i,job_skillset in enumerate(J):
        if all(v == 0 for v in job_skillset):
            remove_list.append(i)
     
    remove_list = sorted(remove_list, reverse=True)
    for i in remove_list:
        # del J[i]
        J = np.delete(J, i, 0)
        
        
    for i in range(0,P.shape[0]):
        for j in range(0,P.shape[1]):
            if (P[i,j] == 1) and (j+1 not in proj_skills_existence):
                P[i,j] = 0
    
    remove_list = []
    for i,job_skillset in enumerate(P):
        if all(v == 0 for v in job_skillset):
            remove_list.append(i)
    
    remove_list = sorted(remove_list, reverse=True)
    for i in remove_list:
        # del P[i]
        P = np.delete(P, i, 0)
    
    return J,P

def dictionary(P):
    P_dict = {}
    for idx,person in enumerate(P):
        P_dict[idx] = person
        
    return P_dict

def list_enumeration(L):
    L_sol = []
    for idx,el in enumerate(L):
        entry = (idx,el)
        L_sol.append(entry)
        
    return L_sol


# what is the maximum number of Q that any of p participates in
def max_load(Q):
    L = float("inf")
    empty = [x for x in Q if x != []]
    if len(empty) == 0:
        return 0
    else:
        x = []
        for el in Q:
            if el:
                for e in el:
                    x.append(e[0])

        count = Counter(x)
        L = count.most_common()[0][1]
        max_person = count.most_common()[0][1]
        return L
    
# Coverage is defined as: C(Q) = F(J1) + F(J2) + ... + F(Jk)
# computation of coverage
def non_coverage(J,J_orig,Q,P_idx):
    
    C = 0; not_covered = 0

    for idx,task in J.iteritems():
        if task:
            not_covered = 0
            team = Q[idx] 
            team_skills = []
            for member in team:
                team_skills+=P_idx[member[0]]

            for req in task:
                if req not in team_skills:
                    not_covered+=1

            F = not_covered/float(len(J_orig[idx]))
            C+=F       
            
            
    return C
    
    
def create_sparse_input_LHS(Ad,Al,Ac):
    '''
    You need three lists:
    1) list of entries of the matrix, with for each entry the value
    2) row index
    3) column index.
    '''
    Ad_nonzero = np.nonzero(Ad)
    Ad_rows = Ad_nonzero[0]
    Ad_columns = Ad_nonzero[1]
    Ad_entries = Ad[Ad_nonzero]
    
    Al_nonzero = np.nonzero(Al)
    Al_rows = Al_nonzero[0]
    Al_rows = [i+Ad.shape[0] for i in Al_rows]
    Al_columns = Al_nonzero[1]
    Al_entries = Al[Al_nonzero]
    
    shape = Ad.shape[0] + Al.shape[0]
    
    Ac_nonzero = np.nonzero(Ac)
    Ac_rows = Ac_nonzero[0]
    Ac_rows = [i+shape for i in Ac_rows]
    Ac_columns = Ac_nonzero[1]
    Ac_entries = Ac[Ac_nonzero]
    
    entries = np.concatenate((Ad_entries, Al_entries, Ac_entries), axis=0)
    row_index = np.concatenate((Ad_rows, Al_rows, Ac_rows), axis=0)
    column_index = np.concatenate((Ad_columns, Al_columns, Ac_columns), axis=0)
    return entries, row_index, column_index


def create_sparse_coefficients(c):
    c_nonzero = np.nonzero(c)
    c_rows = c_nonzero[0]
    c_columns = [0]*len(c_nonzero[0])
    c_entries = c[c_nonzero]
    
    return c_entries, c_rows, c_columns


def create_sparse_input_RHS(bd,bl,bc):
    '''
    You need three lists:
    1) list of entries of the matrix, with for each entry the value
    2) row index
    3) column index.
    '''
    bd_nonzero = np.nonzero(bd)
    bd_rows = bd_nonzero[0]
    bd_columns = [0]*len(bd_nonzero[0])
    bd_entries = bd[bd_nonzero]
    
    bl_nonzero = np.nonzero(bl)
    bl_rows = [0]*len(bl_nonzero[0])
    bl_rows = [i+bd.shape[0] for i in bl_rows]
    bl_columns = [0]*len(bl_nonzero[0])
    bl_entries = bl[bl_nonzero]
    
    shape = bd.shape[0] + bl.shape[0]
    print 'shapeee:',shape
    bc_nonzero = np.nonzero(bc)
    bc_rows = bc_nonzero[0]
    bc_rows = [i+shape for i in bc_rows]
    bc_columns = [0]*len(bc_nonzero[0])
    bc_entries = bc[bc_nonzero]
    
    print 'bd rows:',bd_rows
    print 'bl rows:',bl_rows
    print 'bc rows:',bc_rows
    
    entries = np.concatenate((bd_entries, bl_entries, bc_entries), axis=0)
    row_index = np.concatenate((bd_rows, bl_rows, bc_rows), axis=0)
    column_index = np.concatenate((bd_columns, bl_columns, bc_columns), axis=0)
    print 'entries:',entries
    print 'row index:',row_index
    print 'column index:',column_index
    return entries, row_index, column_index

def cvxopt_solve_minmax(c,Ad,Al,Ac,bd,bl,bc,solver,n):
    
    c = matrix(c)
    (A_entries, A_row_index, A_column_index) = create_sparse_input_LHS(Ad,Al,Ac)
    # A = matrix(vstack([Ad, Al, Ac]))
    A = spmatrix(A_entries, A_row_index, A_column_index)
#     print 'A entries:',A_entries
#     print 'A row index:',A_row_index
#     print 'A column index:',A_column_index
    (b_entries, b_row_index, b_column_index) = create_sparse_input_RHS(bd,bl,bc)
    # b = matrix(hstack([b_entries, b_row_index, b_column_index]))
#     print 'bd:',bd,'bl:',bl,'bc:',bc
#     print 'b entries:',b_entries
#     print 'b row index:',b_row_index
#     print 'b column index:',b_column_index
    b = matrix(hstack([bd, bl, bc]))
    print 'length of b:',len(b)
    sol = solvers.lp(c, A, b, solver=solver, options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
    return sol['x']

def pulp_solve_minmax(J,P, offset = []):
    """
    minimize personal loads.

    Args:
        J: An array with shape (n_tasks,n_features) containing info about tasks
        P: An array with shape (n_people,n_features) containing info about people
        offset: a list of index, indicating people that are not participating (for test purpose)
        
    Returns:
        A binary matrix X with shape (n_people, n_tasks), indicating the assignments.

    """
    # assignment matrix
    X = [[0 for x in range(len(J))] for y in range(len(P))]  

    # declare your variables
    L = LpVariable("L", 0, 1000)
    namestr = 0 
    for i in range(len(X)):
        for j in range(len(X[1])):
            X[i][j] = LpVariable('x'+ str(namestr), 0, 1) # 0=<x<=1
            namestr += 1

    # defines the problem
    prob = LpProblem("problem", LpMinimize)
    
    # defines the objective function to minimize
    prob += L
    
    # find able-cover
    able_cover = np.array(P).sum(0)>0
    
    # find needed-cover
    needed_cover = np.array(J).sum(0)>0
      
    # find those cannot be covered
    needed_delete = [x>y for (x,y) in zip(needed_cover,able_cover)]
    needed_delete_mutiplicity = \
            [np.array(J).sum(0)[i]*needed_delete[i] for i in range(len(needed_delete))]
    
    # defines the regular constraints
    for i in range(len(X)):  # all people's loads subject to a uppper bound 
        prob += sum(X[i])<=L
        
    # offset people cannot participate    
    for i in offset:  
        prob += sum(X[i])==0
        
    # all skills in all tasks must be covered 
    for i in range(len(J)): 
        for j in range(len(J[0])):
            if needed_delete[j] == 0 and J[i][j]==1:              
                prob += cross([a[i] for a in X],[row[j] for row in P]) >= J[i][j]    

    # solve the problem
    status = prob.solve(GLPK(msg=0))

    X = output_pulp(X)
    tot = X.shape[0]*X.shape[1]
    zeros = tot - np.count_nonzero(X)
    return X,value(L),zeros

def create_assignment_matrix(J,P):  
    '''
    Solve the relaxed LP for the non-online setting of the Load minimization problem by Anagnostopoulos et al.
    Solution using the python library pulp.
    Input: Experts and Jobs
    Output: Assignment matrix X with decimal numbers in [0,1]
    '''

    k = len(P); m = len(J); l = len(P[0])  
    
    start_time_pulp = time.time()
    X,mload_pulp,zeros_pulp = pulp_solve_minmax(J,P)
    elapsed_time_pulp = time.time() - start_time_pulp
    
    return X,mload_pulp

def assign_experts_to_projects(X,R,thresh):
    '''
    For R rounds assign with probability of assignment matrix X expert Pi to project Jj. If Pi is assigned to Jj
    even in one round, then it will be also assigned at the end.
    Input: Assignment matrix X which is the solution of optimization problem
    Output: Set of teams Q
    '''
    assignments = {}
    rows = X.shape[0]; cols = X.shape[1]
    seed = 1
    random.seed(seed)
    counter = 0
    
    for ro in range(1,R+1):
        print 'Round number:',ro
        for i in range(0,rows):
            for j in range(0,cols):
                # select a number uniformly at random in the range [0,1]
                iid = random.uniform(0,1)
                # probability of assigning expert i to job j
                prob = X[i][j]
                if prob >= thresh:
                    if iid <= prob:
                        counter+=1
                        pair = (i,j)
                        assignments[pair] = 1
                    
        counter = 0
        
    Xsol = np.zeros((rows,cols))
    for key,value in assignments.iteritems():
        exp = key[0]; proj = key[1];
        Xsol[exp][proj] = 1
        
    return Xsol

def transform_to_team(Xsol,P_idx,J_idx,JMatrix):
    ''' 
    Input: Assignments of experts to projects, dictionary of skills of experts
    Output: Of the form 
        Q: [[(1, [1, 3, 6])], [(3, [1, 6]), (0, [3, 4])], [(4, [1, 18])], [(2, [1])]]
    '''
    rows = Xsol.shape[0]; cols = Xsol.shape[1]
    
    Q = [[]]*len(JMatrix);
    
    for j in range(0,cols):
        for i in range(0,rows):
            expert = i; project = j
            assign = Xsol[expert][project]
            if not Q[project]:
                Q[project] = []
                
            if assign == 1:
                skills = P_idx[expert]
                tup = (expert,skills)
                Q[project].append(tup)
                 
    return Q       


def main():
    current_directory = os.getcwd()
    filename_dataset = "upwork_sample"
    file_people = current_directory+"/datasets/real/"+filename_dataset+"_experts.csv"
    file_tasks = current_directory+"/datasets/real/"+filename_dataset+"_projects.csv"

    (JMatrix,PMatrix) = create_people_tasks(file_people,file_tasks)

    print 'Number of projects initially:',len(JMatrix)
    print 'Number of experts initially:',len(PMatrix)
    print 
    # removing overlapping
    (JDL,PD) = convert_data(JMatrix,PMatrix)
    (exp_skills_existence,proj_skills_existence) = expert_project_skills(PD,JDL)

    (JMatrix,PMatrix) = remove_nonoverlap_skils(JMatrix,PMatrix,exp_skills_existence,proj_skills_existence)

    print 'Number of projects after removing non overlapping:',len(JMatrix)
    print 'Number of experts after removing non overlapping:',len(PMatrix)

    # converting to dictionary
    (JDL,PD) = convert_data(JMatrix,PMatrix)

    # dictionary with person id and skills
    P_idx = dictionary(PD)
    J_idx = dictionary(JDL)


    # Solving the optimization problem problem
    m = len(JMatrix); k = len(PMatrix); l = len(PMatrix[0])

    start_time_assignment = time.time()
    (X,mload) = create_assignment_matrix(JMatrix,PMatrix)
    end_time_assignment = time.time() - start_time_assignment
    print 'Assignment time took:',end_time_assignment
    np.savetxt(current_directory+"/assignment_matrices/assignment_X_guru.csv", X, delimiter=",")
    print 'Solved X'
    
    
if __name__== "__main__":
    main()