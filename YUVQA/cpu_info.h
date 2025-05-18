#ifndef CUP_INFO
#define CUP_INFO

#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
int get_physical_cores_windows() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buffer(len);
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX ptr =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, ptr, &len)) {
        return -1;
    }

    int coreCount = 0;
    DWORD offset = 0;
    while (offset < len) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (info->Relationship == RelationProcessorCore)
            ++coreCount;
        offset += info->Size;
    }

    return coreCount;
}
#else

#include <fstream>
#include <string>
#include <set>
#include <iostream>

int get_physical_cores_unix() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return -1; // error
    }

    std::string line;
    std::set<std::pair<int, int>> cores; // (physical id, core id)

    int physical_id = -1;
    int core_id = -1;

    while (std::getline(cpuinfo, line)) {
        if (line.find("physical id") != std::string::npos) {
            physical_id = std::stoi(line.substr(line.find(":") + 1));
        }
        else if (line.find("core id") != std::string::npos) {
            core_id = std::stoi(line.substr(line.find(":") + 1));
        }
        else if (line.empty()) {
            if (physical_id != -1 && core_id != -1) {
                cores.insert({ physical_id, core_id });
            }
            physical_id = core_id = -1;
        }
    }

    return cores.size();
}

#endif



int get_physical_cores() {
#ifdef _WIN32
    return get_physical_cores_windows();
#else
#if defined(__linux__) || defined(__APPLE__)
    return get_physical_cores_unix();
#else
    return -1; // Unknown platform
#endif
#endif
}


#endif // CUP_INFO
