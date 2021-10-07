ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=2 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=0
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order2_2x1x10.dat
ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=2 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=1
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order2_4x2x20.dat
ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=2 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=2
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order2_8x4x40.dat
ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=3 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=0
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order3_2x1x10.dat
ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=3 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=1
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order3_4x2x20.dat
ninja SpaceTimeVlasovPoisson && time ./drivers/experimental/TwoStreamInstability/SpaceTimeVlasovPoisson --H1Order=3 --xElements=2 --tElements=1 --vElements=10 --problem=cold-diode --viz --numSubsteps=5 --numUniformRefinements=2
python compare_exact_Jaffe_values.py --inputFile ./cold-diode_results_L2Trace_u01_H1Order3_8x4x40.dat
