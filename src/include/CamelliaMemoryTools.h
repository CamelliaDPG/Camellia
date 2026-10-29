//
//  CamelliaMemoryTools.h
//  Camellia
//
//  Created by Roberts, Nathan V on 8/17/20.
//

#ifndef CamelliaMemoryTools_h
#define CamelliaMemoryTools_h

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace Camellia
{

  inline double memoryHighWaterMarkInMegabytes()
  {
    struct rusage sys_resources;
    getrusage(RUSAGE_SELF, &sys_resources);

    // sys_resources.ru_maxrss is memory high-water mark in KB; divide by 1024 for MB.
    double usage_in_MB = double(sys_resources.ru_maxrss)/1024.0;

    // MacOS is off by a factor of 1024 (appears to report in bytes)
  #if __APPLE__
    usage_in_MB /= 1024.0;
  #endif
    return usage_in_MB;
  }

  inline double memoryHighWaterMarkInGigabytes()
  {
    return memoryHighWaterMarkInMegabytes() / 1024.0;
  }

  //! current resident set size of this process, in GB (0 if unavailable)
  inline double memoryCurrentResidentInGigabytes()
  {
  #if __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return double(info.resident_size) / (1024.0 * 1024.0 * 1024.0);
  #else
    FILE *statm = fopen("/proc/self/statm", "r");
    if (statm == NULL) return 0;
    long totalPages = 0, residentPages = 0;
    const int numRead = fscanf(statm, "%ld %ld", &totalPages, &residentPages);
    fclose(statm);
    if (numRead != 2) return 0;
    return double(residentPages) * double(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0 * 1024.0);
  #endif
  }

  //! System-wide memory pressure.  On macOS, the kernel's memory-pressure level (kern.memorystatus_vm_pressure_level:
  //! 1 = normal, 2 = warning, 4 = critical; the warning/critical states are what precede the OS's out-of-memory warnings and,
  //! under sustained pressure, compressor exhaustion).  On Linux, the percentage of the last 10 seconds in which some task
  //! stalled on memory (PSI, /proc/pressure/memory "some avg10").  Returns a negative value if unavailable.
  inline double systemMemoryPressure()
  {
  #if __APPLE__
    int level = 0;
    size_t length = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &length, NULL, 0) != 0) return -1;
    return double(level);
  #else
    FILE *psi = fopen("/proc/pressure/memory", "r");
    if (psi == NULL) return -1;
    double someAvg10 = -1;
    const int numRead = fscanf(psi, "some avg10=%lf", &someAvg10);
    fclose(psi);
    return (numRead == 1) ? someAvg10 : -1;
  #endif
  }

  //! A named memory checkpoint: current resident size and high-water mark of this process when the checkpoint was recorded.
  struct MemoryCheckpoint
  {
    std::string name;
    double currentGB;
    double highWaterMarkGB;
  };

  //! Process-wide log of memory checkpoints, recorded at phase boundaries (e.g., within Solution::solve()).  Recording is
  //! off unless enabled; a driver enables it, then reads and clears the log after each solve.
  //!
  //! If the environment variable CAMELLIA_MEMORY_LIMIT_GB is set, each checkpoint (whether or not recording is enabled)
  //! aborts the process when its current resident size exceeds that many GB, so that runs that would exhaust system memory
  //! fail early instead.  The limit applies per process (per MPI rank).
  //!
  //! Because per-process limits cannot see system-wide memory exhaustion (other processes, or the OS's compressed memory
  //! running out of segments), CAMELLIA_MEMORY_PRESSURE_ABORT additionally aborts when system-wide memory pressure is high:
  //! - macOS: "critical" (default if the variable is set to anything else non-numeric) or "warning", compared with the
  //!   kernel's memory-pressure level;
  //! - Linux: a number, the PSI memory stall percentage (avg10) at or above which to abort (e.g., 20); "warning" and
  //!   "critical" mean 10 and 20.
  //! On macOS a numeric value is interpreted as the pressure level (2 = warning, 4 = critical).
  class MemoryCheckpoints
  {
    static bool &enabledFlag() { static bool enabled = false; return enabled; }
    static std::vector<MemoryCheckpoint> &log() { static std::vector<MemoryCheckpoint> checkpoints; return checkpoints; }
    static double limitGB()
    {
      static const double limit = []() {
        const char *limitString = getenv("CAMELLIA_MEMORY_LIMIT_GB");
        return (limitString == NULL) ? -1.0 : atof(limitString);
      }();
      return limit;
    }
    //! threshold for systemMemoryPressure() at or above which to abort; negative if the check is disabled
    static double pressureThreshold()
    {
      static const double threshold = []() {
        const char *value = getenv("CAMELLIA_MEMORY_PRESSURE_ABORT");
        if (value == NULL) return -1.0;
        const std::string setting(value);
      #if __APPLE__
        if (setting == "warning") return 2.0;
        if (setting == "critical") return 4.0;
      #else
        if (setting == "warning") return 10.0;
        if (setting == "critical") return 20.0;
      #endif
        char *end = NULL;
        const double number = strtod(value, &end);
        if ((end != value) && (*end == '\0')) return number;
      #if __APPLE__
        return 4.0;
      #else
        return 20.0;
      #endif
      }();
      return threshold;
    }
    static void abortIfUnderPressure(const std::string &name)
    {
      const double threshold = pressureThreshold();
      if (threshold < 0) return;
      const double pressure = systemMemoryPressure();
      if (pressure >= threshold)
      {
        std::cerr << "Camellia: system memory pressure " << pressure << " at checkpoint '" << name
                  << "' is at or above CAMELLIA_MEMORY_PRESSURE_ABORT threshold " << threshold << "; aborting.\n";
        std::cerr.flush();
        abort();
      }
    }
    static void abortIfOverLimit(const std::string &name, const double current, const double limit)
    {
      if ((limit > 0) && (current > limit))
      {
        std::cerr << "Camellia: resident memory " << current << " GB at checkpoint '" << name
                  << "' exceeds CAMELLIA_MEMORY_LIMIT_GB = " << limit << "; aborting.\n";
        std::cerr.flush();
        abort();
      }
    }
  public:
    static void setEnabled(bool value) { enabledFlag() = value; }
    static bool enabled() { return enabledFlag(); }

    static void record(const std::string &name)
    {
      abortIfUnderPressure(name);
      const double limit = limitGB();
      if (!enabledFlag() && (limit <= 0)) return;
      const double current = memoryCurrentResidentInGigabytes();
      if (enabledFlag()) log().push_back({name, current, memoryHighWaterMarkInGigabytes()});
      abortIfOverLimit(name, current, limit);
    }

    //! enforces CAMELLIA_MEMORY_LIMIT_GB and CAMELLIA_MEMORY_PRESSURE_ABORT (if set) without recording a checkpoint;
    //! suitable for use inside loops
    static void checkLimit(const std::string &name)
    {
      abortIfUnderPressure(name);
      const double limit = limitGB();
      if (limit <= 0) return;
      abortIfOverLimit(name, memoryCurrentResidentInGigabytes(), limit);
    }

    static const std::vector<MemoryCheckpoint> &checkpoints() { return log(); }
    static void clear() { log().clear(); }
  };
}

#endif /* CamelliaMemoryTools_h */
