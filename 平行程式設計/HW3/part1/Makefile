SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG
PROGRAMNAME=cg
DATASIZE=MEDIUMN
# By now, we are only using medium-sized data

default: ${PROGRAMNAME} grade

include make.common

OBJS = cg_impl.o \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o

${PROGRAMNAME}: config ${PROGRAMNAME}.o ${OBJS}
	${CLINK} ${CLINKFLAGS} -Wl,--allow-multiple-definition -g -o ${PROGRAMNAME} ${PROGRAMNAME}.o ${OBJS} ${C_LIB}

grade: config grade.o ${OBJS}
	${CLINK} ${CLINKFLAGS} -Wl,--allow-multiple-definition -g -o cg_grader grade.o ${OBJS} ref_cg.a def_cg.a ${C_LIB}

.c.o:
	${CCOMPILE} $< -D${DATASIZE}

cg.o:	cg.c  globals.h
cg_impl.o:	cg_impl.c  globals.h

clean:
	- rm -f *.o *~
	rm -f ${COMMON}/*.o
	rm -f ${PROGRAMNAME} cg_grader
