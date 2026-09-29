/* Bounded ABI-1 command submission. Cache maintenance and ownership stay in
 * Python; volatile MMIO and full fences order device accesses. No /dev/mem,
 * allocation, FPGA reconfiguration, or implicit retry occurs in this library.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#include <time.h>
#define EXPORT __attribute__((visibility("default")))
#endif

/* Test builds substitute register and clock operations only. The same command
 * and deadline code runs against a separately implemented device model. */
#ifdef PLD_CONTROL_TEST
static uint32_t (*test_read)(uintptr_t, uint32_t);
static void (*test_write)(uintptr_t, uint32_t, uint32_t);
static uint64_t (*test_clock)(void);
EXPORT void control_test_hooks(uint32_t (*r)(uintptr_t,uint32_t),
                               void (*w)(uintptr_t,uint32_t,uint32_t),
                               uint64_t (*c)(void)) {
    test_read=r; test_write=w; test_clock=c;
}
#endif

static uint32_t rd(volatile uint32_t *base, uint32_t offset) {
#ifdef PLD_CONTROL_TEST
    return test_read((uintptr_t)base,offset);
#else
    return base[offset/4];
#endif
}
static void wr(volatile uint32_t *base, uint32_t offset, uint32_t value) {
#ifdef PLD_CONTROL_TEST
    test_write((uintptr_t)base,offset,value);
#else
    base[offset/4]=value;
#endif
}
static uint64_t now_ns(void) {
#ifdef PLD_CONTROL_TEST
    return test_clock();
#elif defined(_WIN32)
    LARGE_INTEGER ticks,frequency;
    if (!QueryPerformanceCounter(&ticks) || !QueryPerformanceFrequency(&frequency)) return 0;
    return (uint64_t)(ticks.QuadPart/frequency.QuadPart)*1000000000ULL +
           (uint64_t)(ticks.QuadPart%frequency.QuadPart)*1000000000ULL/(uint64_t)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC,&value)) return 0;
    return (uint64_t)value.tv_sec*1000000000ULL+(uint64_t)value.tv_nsec;
#endif
}
static int valid_region(const volatile uint32_t *p, uint32_t bytes) {
    return p && !((uintptr_t)p & 3) && bytes>=0x5c;
}
static int expired(uint64_t start, uint64_t timeout) {
    uint64_t now=now_ns();
    if (!now || now<start) return -1;
    return now-start>=timeout;
}

/* Return codes: 1 arguments, 2 busy/faulted core, 3 halted DMA before launch,
 * 4 DMA error, 5 unexpected DMA halt, 6 DMA timeout, 7 RX length,
 * 8 core error, 9 core timeout, 10 stale completion/length, 11 clock failure.
 * result: command cycles, compute cycles, core status, DMA status,
 * DMA nanoseconds low/high, core error, reserved.
 */
EXPORT uint32_t tensor_control_abi(void) { return 1; }

EXPORT int tensor_configure(volatile uint32_t *core, uint32_t core_bytes,
                            const uint32_t values[8]) {
    if (!valid_region(core,core_bytes) || !values) return 1;
    if (rd(core,0x0c)&0x45) return 2;
    for (unsigned i=0;i<8;++i) wr(core,0x18+4*i,values[i]);
    __sync_synchronize();
    return 0;
}

static int wait_dma(volatile uint32_t *core, volatile uint32_t *dma, unsigned base,
                    uint32_t length, uint64_t timeout, uint32_t result[8]) {
    uint64_t start=now_ns();
    if (!start) return 11;
    for (;;) {
        uint32_t status=rd(dma,base+4);
        result[3]=status;
        if (status&0x70) return 4;
        if (status&1) return 5;
        if ((status&0x1002)==0x1002) break;
        result[2]=rd(core,0x0c);
        if (result[2]&4) { result[6]=rd(core,0x10); return 8; }
        int timeout_state=expired(start,timeout);
        if (timeout_state) return timeout_state<0 ? 11 : 6;
    }
    __sync_synchronize();
    if (base && rd(dma,0x58)!=length) return 7;
    return 0;
}

static int wait_core(volatile uint32_t *core, uint32_t command, uint32_t frame,
                    uint32_t job, uint32_t length, uint64_t timeout, uint32_t result[8]) {
    uint64_t start=now_ns();
    if (!start) return 11;
    for (;;) {
        uint32_t status=rd(core,0x0c);
        result[2]=status;
        if (status&4) { result[6]=rd(core,0x10); return 8; }
        if (status&2) {
            if ((status&1) || rd(core,0x3c)!=frame || rd(core,0x40)!=job) return 10;
            uint32_t received=command==4 ? 0 : length;
            uint32_t sent=command==4 ? length : 0;
            if (rd(core,0x50)!=received || rd(core,0x54)!=sent ||
                rd(core,command==4 ? 0x4c : 0x48)!=length) return 10;
            __sync_synchronize();
            result[0]=rd(core,0x44); result[1]=rd(core,0x58);
            return 0;
        }
        int timeout_state=expired(start,timeout);
        if (timeout_state) return timeout_state<0 ? 11 : 9;
    }
}

EXPORT int tensor_wait_core(volatile uint32_t *core, uint32_t core_bytes,
                            uint32_t command, uint32_t frame, uint32_t job,
                            uint32_t length, uint64_t timeout, uint32_t result[8]) {
    if (!valid_region(core,core_bytes) || !result || command<1 || command>4 ||
        !job || !length || !timeout) return 1;
    for (unsigned i=0;i<8;++i) result[i]=0;
    return wait_core(core,command,frame,job,length,timeout,result);
}

EXPORT int tensor_packet(volatile uint32_t *core, uint32_t core_bytes,
                         volatile uint32_t *dma, uint32_t dma_bytes,
                         uint32_t command, uint32_t frame, uint32_t job,
                         uint64_t address, uint32_t length, uint64_t timeout,
                         uint32_t result[8]) {
    if (!valid_region(core,core_bytes) || !valid_region(dma,dma_bytes) || !result ||
        command<1 || command>4 || !job || !length || length>=0x4000000 ||
        !timeout || (address&7) || address>UINT32_MAX ||
        address+length>0x100000000ULL) return 1;
    for (unsigned i=0;i<8;++i) result[i]=0;
    unsigned base=command==4 ? 0x30 : 0;
    result[2]=rd(core,0x0c);
    if (result[2]&0x45) return 2;
    result[3]=rd(dma,base+4);
    if (result[3]&0x70) return 4;
    if (result[3]&1) return 3;
    uint64_t start=now_ns();
    if (!start) return 11;
    /* All payload validation/cache writeback must have completed before here.
     * Clear the old DMA IOC before LENGTH starts the new transfer. */
    __sync_synchronize();
    wr(core,0x14,command); wr(core,0x38,job);
    __sync_synchronize();
    wr(core,0x08,1);
    wr(dma,base+4,0x7000);
    wr(dma,base+0x18,(uint32_t)address);
    wr(dma,base+0x1c,0);
    __sync_synchronize();
    wr(dma,base+0x28,length);
    __sync_synchronize();
    int status=wait_dma(core,dma,base,length,timeout,result);
    if (status) return status;
    uint64_t end=now_ns();
    if (!end || end<start) return 11;
    uint64_t elapsed=end-start;
    result[4]=(uint32_t)elapsed; result[5]=(uint32_t)(elapsed>>32);
    return wait_core(core,command,frame,job,length,timeout,result);
}
