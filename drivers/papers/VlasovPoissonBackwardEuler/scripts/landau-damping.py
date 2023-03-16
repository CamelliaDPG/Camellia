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

def runLandauDamping(xMeshWidth, vMeshWidth, numUniformRefinements, \
                     dt, numTimeSteps, p, dp):
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
    driverName = "LandauDamping"
    commandPath = buildDir + "/drivers/papers/VlasovPoissonBackwardEuler/" + driverName
    commandArgs  = ["--viz", "--iterativeSolveTolerance=1e-10", "--H1Order=" + p]
    commandArgs.append("--delta_k=" + dp)
    commandArgs.append("--xElements=" + xMeshWidth)
    commandArgs.append("--vElements=" + vMeshWidth)
    commandArgs.append("--numTimeSteps=" + numTimeSteps)
    commandArgs.append("--dt=" + dt)
    commandArgs.append("--iterativeSolveOutputLevel=0") # don't output the results of CG iteration
    commandArgs.append("--outputFrequency=100") # for all experiments, output once every 100 time steps
    commandArgs.append("--checkpointFrequency=10000") # for all experiments, checkpoint once every 10000 time steps
    outputDir += "/landau"
    commandArgs.append("--v0=-10")
    commandArgs.append("--vWidth=20")
    xMeshWidthForOutput = int(xMeshWidth) << int(numUniformRefinements)
    vMeshWidthForOutput = int(vMeshWidth) << int(numUniformRefinements)
    outputDir += "_p" + p + "_dp" + dp + "_" + str(xMeshWidthForOutput) + "x"
    outputDir += str(vMeshWidthForOutput)
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
    with open(outputDir + "/" + consoleOutputFileName, "w") as consoleOutputFile:
      consoleOutputFile.write(" ".join(mpiCommand))
      consoleOutputFile.write("\n")
    with open(outputDir + "/" + consoleOutputFileName, "a") as consoleOutputFile:
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
    parser.add_option("--xMeshWidth", action="store", type="string", dest="xMeshWidth")
    parser.add_option("--vMeshWidth", action="store", type="string", dest="vMeshWidth")
    parser.add_option("--numUniformRefinements", action="store", type="string", dest="numUniformRefinements", default="0")
    parser.add_option("--dt", action="store", type="string", dest="dt", default="0.001")
    parser.add_option("--numTimeSteps", action="store", type="string", dest="numTimeSteps", default="80000")
        
    parser.add_option("--p",  action="store", type="string", dest="p", default="1")  
    parser.add_option("--dp", action="store", type="string", dest="dp", default="2")
    
    parser.add_option("--overrideRepoCheck", action="store_true", dest="overrideRepoCheck", default=False)

    opt, args = parser.parse_args()
    if not opt.xMeshWidth:
        parser.error("xMeshWidth argument is required")
    if not opt.vMeshWidth:
        parser.error("vMeshWidth argument is required")
    
    if (not opt.overrideRepoCheck and not repoIsClean()):
        print("Repo is dirty; exiting")
        sys.exit(1)
    
    runLandauDamping(opt.xMeshWidth, opt.vMeshWidth, opt.numUniformRefinements, \
                     opt.dt, opt.numTimeSteps, opt.p, opt.dp)

