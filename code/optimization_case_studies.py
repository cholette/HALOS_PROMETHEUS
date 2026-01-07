## %% Optimize case studies
import time
import inputs
import optimize_aimpoint
import process_aimpoint_outputs

case_filename = "./../case_inputs/flat_50_ca_case.csv"
if __name__ == "__main__":
    t_start= time.time()
    fm = inputs.getFullFluxModelFromFiles(case_filename)
    ao_params = {"section_id":1,"num_sections":1,"hour_idx":fm.settings["hour_idx"],"aimpoint_cons_only":False}
    ao = optimize_aimpoint.AimpointOptimizer(fm,ao_params)
    elapsed_1 = time.time()-t_start
    print("time to set up objects: ",elapsed_1)
    t_start= time.time()
    ao.createFullProblem()
    elapsed_2 = time.time()-t_start
    print("time to build model: ",elapsed_2)
    t_start= time.time()
    ao.optimize()
    elapsed_3 = time.time()-t_start
    print("time to solve:",elapsed_3)
    t_start = time.time()
    outputs = process_aimpoint_outputs.AimpointOptOutputs(ao,fm)
    elapsed_4 = time.time() - t_start
    print("Output processing time: ",elapsed_4)
    build_and_solve_time = elapsed_1+elapsed_2
    print("Total computing time: ",(elapsed_1+elapsed_2+elapsed_3))
    outputs.setSolveTime(build_and_solve_time)
