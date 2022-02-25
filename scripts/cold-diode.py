#!/usr/local/bin/python3

from git import *
import optparse
import os
import subprocess
import sys
import time

def repoHash():
    repoPath = os.environ.get("CAMELLIA_SRC_DIR")
    if (repoPath is None):
        print("CAMELLIA_SRC_DIR environment variable must be set")
        sys.exit(1)
    repo = Repo(repoPath)
    sha = repo.head.commit.hexsha
    short_sha = repo.git.rev_parse(sha, short=4)
    return short_sha
    
def repoBranch():
    repoPath = os.environ.get("CAMELLIA_SRC_DIR")
    if (repoPath is None):
        print("CAMELLIA_SRC_DIR environment variable must be set")
        sys.exit(1)
    repo = Repo(repoPath)
    return repo.active_branch.name

def repoIsClean():
    repoPath = os.environ.get("CAMELLIA_SRC_DIR")
    if (repoPath is None):
        print("CAMELLIA_SRC_DIR environment variable must be set")
        sys.exit(1)
    repo = Repo(repoPath)
    uncommitted = repo.is_dirty()
    return not repo.is_dirty()

def printRepoStatus(repoPath):
    repo = Repo(repoPath)
    lastCommit = repo.head.commit.committed_date
    uncommitted = repo.is_dirty()
    branch = repo.active_branch
    print("On branch: " + branch.name)
    sha = repo.head.commit.hexsha
    short_sha = repo.git.rev_parse(sha, short=4)
    print("commit: " + short_sha)
    if (uncommitted):
        print("Repo is dirty")
    else:
        print("Repo is clean")

def runColdDiode(spaceTime, adaptive, refinementCount, energyThreshold, bcTol, \
                 xMeshWidth, tMeshWidth, vMeshWidth, numUniformRefinements, \
                 numTimeSteps, timeRamp, sigma, p, dp, v0, vWidth, maxElements):
    buildDir = os.environ.get("CAMELLIA_BUILD_DIR")
    if (buildDir is None):
        print("CAMELLIA_BUILD_DIR environment variable must be set")
        sys.exit(1)
    outputDir = os.environ.get("CAMELLIA_OUTPUT_DIR")
    if (outputDir is None):
        print("CAMELLIA_OUTPUT_DIR environment variable must be set")
        sys.exit(1)
    #mpiPath = "/usr/local/bin/mpirun"
    mpiPath = os.environ.get("MPIRUN_COMMAND")
    if (mpiPath is None):
        print("MPIRUN_COMMAND environment variable must be set")
        sys.exit(1)
    mpiNumProcsString = os.environ.get("MPIRUN_NUM_PROCS")
    if (mpiNumProcsString is None):
        print("MPIRUN_NUM_PROCS environment variable must be set")
        sys.exit(1)
    mpiNumProcs = int(mpiNumProcsString)
    driverName = ""
    if spaceTime:
        driverName = "SpaceTimeVlasovPoisson"
    else:
        driverName = "TwoStreamInstability"
    commandPath = buildDir + "/drivers/experimental/TwoStreamInstability/" + driverName
    commandArgs  = ["--problem=cold-diode", "--viz", "--fixedPointTol=1e-6", "--iterativeSolveTolerance=1e-9", "--H1Order=" + p]
    commandArgs.append("--delta_k=" + dp)
    commandArgs.append("--xElements=" + xMeshWidth)
    if spaceTime:
      commandArgs.append("--tElements=" + tMeshWidth)
    commandArgs.append("--vElements=" + vMeshWidth)
    if not spaceTime:
        commandArgs.append("--numTimeSteps=" + numTimeSteps)
        commandArgs.append("--dt=" + str(1.0 / float(numTimeSteps)))
        commandArgs.append("--dontGateRefinements")
    commandArgs.append("--sigma=" + sigma)
    if spaceTime:
        driverName = "SpaceTimeVlasovPoisson"
        outputDir += "/space-time"
    else:
        driverName = "TwoStreamInstability"
        outputDir += "/time-marching"
    if adaptive:
        commandArgs.append("--useAdaptivity")
        commandArgs.append("--refinementThreshold=" + energyThreshold)
        if spaceTime:
            commandArgs.append("--numRefinements=" + refinementCount)
        if (maxElements is not None):
          commandArgs.append("--maxElements=" + maxElements)
        commandArgs.append("--bcResolutionTolerance=" + bcTol)
        outputDir += "/bcTol" + bcTol
    else:
        commandArgs.append("--useFixedMesh")
        outputDir += "/fixed_mesh"
    commandArgs.append("--v0=" + v0)
    commandArgs.append("--vWidth=" + vWidth)
    xMeshWidthForOutput = int(xMeshWidth) << int(numUniformRefinements)
    if spaceTime:
        if float(timeRamp) > 0:
            tMeshWidthForOutput = (int(tMeshWidth) + 1) << int(numUniformRefinements)
        else:
            tMeshWidthForOutput = int(tMeshWidth) << int(numUniformRefinements)
    vMeshWidthForOutput = int(vMeshWidth) << int(numUniformRefinements)
    commandArgs.append("--sigma=" + sigma)
    outputDir += "/sigma" + sigma + "_p" + p + "_dp" + dp + "_" + str(xMeshWidthForOutput) + "x"
    if spaceTime:
        outputDir += str(tMeshWidthForOutput) + "x"
    outputDir += str(vMeshWidthForOutput)
    if spaceTime:
        outputDir += "_timeRamp" + timeRamp
        commandArgs.append("--initialTimeDivision=" + timeRamp)
    if adaptive:
        outputDir += "_AMR_theta" + energyThreshold + "_" + refinementCount + "_refs"
        if (maxElements is not None):
          outputDir += "_maxElements_" + maxElements
        #sigma0.025_p3_dp4_4x10x40_timeRamp0_AMR_theta0.2_4_refs
    commandArgs.append("--numUniformRefinements=" + numUniformRefinements)
    commandArgs.append("--outputDir=" + outputDir)
    os.chdir(buildDir)
    mkdirProcess = subprocess.run(["mkdir", "-p", outputDir])
    if (mkdirProcess.returncode != 0):
      print("failed to create directory" + outputDir + "; exiting.")
      sys.exit(1)
    buildProcess = subprocess.run(["ninja", driverName])
    if (buildProcess.returncode != 0):
        print("build failed for driver " + driverName + "; exiting.")
        sys.exit(1)
    fullCommand = commandArgs
    fullCommand.insert(0,commandPath)
    fullCommandString = " ".join(fullCommand)
    mpiCommand = [mpiPath,"--np",str(mpiNumProcs)]
    mpiCommand.extend(fullCommand)
    #print(commandPath)
    #print(commandArgs)
    #print(fullCommand)
    print("Running:\n" + " ".join(mpiCommand))
    startTime = time.time()
    consoleOutputFileName = "console-" + repoBranch() + "-" + repoHash() + ".txt"
    consoleOutputFile = open(outputDir + "/" + consoleOutputFileName, "w")
    runProcess = subprocess.run(mpiCommand,stdout=consoleOutputFile,stderr=consoleOutputFile)
    endTime = time.time()
    runTime = (endTime - startTime)
    if (runProcess.returncode != 0):
        print("process returned error code " + str(runProcess.returncode) + "; exiting.")
        sys.exit(1)
    hours   = int(runTime / 3600)
    minutes = int(runTime / 60) - hours * 60
    seconds = int(runTime - minutes * 60 - hours * 3600)
    timingString = "Completed in "
    if hours > 0:
      timingString += str(hours) + " hours, "
    if minutes > 0:
      timingString += str(minutes) + " minutes, "
    timingString += str(seconds) + " seconds.\n"
    print(timingString)
    consoleOutputFile.write(timingString)

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--spaceTime", action="store_true", dest="spaceTime")
    parser.add_option("--timeMarching", action="store_false", dest="spaceTime")
    
    parser.add_option("--adaptive", action="store_true", dest="useAdaptivity")
    parser.add_option("--fixedMesh", action="store_false", dest="useAdaptivity")
    
    parser.add_option("--maxElements", action="store", dest="maxElements")
    
    parser.add_option("--refinementCount", action="store", type="string", dest="numRefinements", default="-1") # non-negative value must be provided for adaptive runs
    parser.add_option("--energyThreshold", action="store", type="string", dest="energyThreshold", default="0.0")
    parser.add_option("--bcTol", action="store", type="string", dest="bcTol", default="1.0") # 1.0 means no BC-related refinement will take place
    
    parser.add_option("--xMeshWidth", action="store", type="string", dest="xMeshWidth")
    parser.add_option("--tMeshWidth", action="store", type="string", dest="tMeshWidth")
    parser.add_option("--vMeshWidth", action="store", type="string", dest="vMeshWidth")
    parser.add_option("--numUniformRefinements", action="store", type="string", dest="numUniformRefinements", default="0")
    parser.add_option("--numTimeSteps", action="store", type="string", dest="numTimeSteps", default="-1")  # positive value must be provided for time-marching
    parser.add_option("--timeRamp", action="store", type="string", dest="timeRamp", default="-1.0")      # non-negative value must be provided for space-time
    
    parser.add_option("--sigma", action="store", type="string", dest="sigma", default="-1")   # non-negative value must be provided
    
    parser.add_option("--p",  action="store", type="string", dest="p", default="0")                        # positive value must be provided
    parser.add_option("--dp", action="store", type="string", dest="dp", default="-1")                      # non-negative value must be provided
    parser.add_option("--v0", action="store", type="string", dest="v0", default="0.5")
    parser.add_option("--vWidth", action="store", type="string", dest="vWidth", default="1.0")

    opt, args = parser.parse_args()
    if not opt.xMeshWidth:
        parser.error("xMeshWidth argument is required")
    if not opt.vMeshWidth:
        parser.error("vMeshWidth argument is required")
    
    if (not repoIsClean()):
        print("Repo is dirty; exiting")
        sys.exit(1)
    
    runColdDiode(opt.spaceTime, opt.useAdaptivity, opt.numRefinements, opt.energyThreshold, opt.bcTol, \
                 opt.xMeshWidth, opt.tMeshWidth, opt.vMeshWidth, opt.numUniformRefinements, \
                 opt.numTimeSteps, opt.timeRamp, opt.sigma, opt.p, opt.dp, opt.v0, opt.vWidth, opt.maxElements)

    
