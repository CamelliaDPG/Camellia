ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=2 --xElements=2 --vElements=10 --numTimeSteps=1 --dt=1.0 --numUniformRefinements=0
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt1_H1Order2_2x10.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=2 --xElements=2 --vElements=10 --numTimeSteps=2 --dt=0.5 --numUniformRefinements=1
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.5_H1Order2_4x20.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=2 --xElements=2 --vElements=10 --numTimeSteps=4 --dt=0.25 --numUniformRefinements=2
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.25_H1Order2_8x40.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=2 --xElements=2 --vElements=10 --numTimeSteps=8 --dt=0.125 --numUniformRefinements=3
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.125_H1Order2_16x80.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=3 --xElements=2 --vElements=10 --numTimeSteps=1 --dt=1.0 --numUniformRefinements=0
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt1_H1Order3_2x10.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=3 --xElements=2 --vElements=10 --numTimeSteps=2 --dt=0.5 --numUniformRefinements=1
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.5_H1Order3_4x20.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=3 --xElements=2 --vElements=10 --numTimeSteps=4 --dt=0.25 --numUniformRefinements=2
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.25_H1Order3_8x40.dat
ninja TwoStreamInstability && time ./drivers/experimental/TwoStreamInstability/TwoStreamInstability --problem="cold-diode" --viz --outputDir="." --outputFrequency=1 --H1Order=3 --xElements=2 --vElements=10 --numTimeSteps=8 --dt=0.125 --numUniformRefinements=3
python compare_exact_Jaffe_values.py --inputFile ./ColdDiode_resultsImplicit_L2Trace_u01_dt0.125_H1Order3_16x80.dat

