// =============================================================================
//  Program : spi.h
//  Author  : CVA6 project
//  Date    : 
// -----------------------------------------------------------------------------
//  Description:
//
//  These are the functions for accessing Xilinx AXI quad spi IP.  This file was
//  retrieved from the Araine project, under: araine/fpga/src/bootrom/src/spi.c
//  The original file has no header that provides the author information.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// -----------------------------------------------------------------------------
//  License information:
//
//  Please check the Ariane project at https://github.com/openhwgroup/cva6.
// =============================================================================
#pragma once

#define SPI_BASE 0xC2000000

#define SPI_RESET_REG SPI_BASE + 0x40
#define SPI_CONTROL_REG SPI_BASE + 0x60
#define SPI_STATUS_REG SPI_BASE + 0x64
#define SPI_TRANSMIT_REG SPI_BASE + 0x68
#define SPI_RECEIVE_REG SPI_BASE + 0x6C
#define SPI_SLAVE_SELECT_REG SPI_BASE + 0x70
#define SPI_TRANSMIT_OCCUPANCY SPI_BASE + 0x74
#define SPI_RECEIVE_OCCUPANCY SPI_BASE + 0x78
#define SPI_INTERRUPT_GLOBAL_ENABLE_REG SPI_BASE + 0x1c
#define SPI_INTERRUPT_STATUS_REG SPI_BASE + 0x20
#define SPI_INTERRUPT_ENABLE_REG SPI_BASE + 0x28

void write_reg(unsigned int addr, unsigned int value);

unsigned int read_reg(unsigned int addr);

void spi_init();

unsigned char spi_txrx(unsigned char byte);

// return -1 if something went wrong
int spi_write_bytes(unsigned char *bytes, unsigned int len, unsigned char *ret);

