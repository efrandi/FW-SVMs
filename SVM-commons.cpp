#include "SVM-commons.h"

#if 1
void info(const char *fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
}
void info_flush()
{
	fflush(stdout);
}
#else
void info(const char *fmt,...) {}
void info_flush() {}
#endif

double getRunTime()
{
#ifdef _MSC_VER
  clock_t current = clock();
  return (double)(current) / CLOCKS_PER_SEC;
#else
  struct tms current;
  times(&current);
  
  double norm = (double)sysconf(_SC_CLK_TCK);
  return(((double)current.tms_utime)/norm);
#endif
}