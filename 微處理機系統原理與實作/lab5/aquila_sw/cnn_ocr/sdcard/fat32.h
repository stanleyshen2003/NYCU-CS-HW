// =============================================================================
//  Program : fat32.h
//  Author  : ChihYu Hsiang 
//  Date    : Oct/29/2019
// -----------------------------------------------------------------------------
//  Description:
//  This is fat32 tool header for aquila bootrom that can parse
//  fat32 header and copy elf and binary code to memory.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// -----------------------------------------------------------------------------
//  License information:
//
//  This software is released under the BSD-3-Clause Licence,
//  see https://opensource.org/licenses/BSD-3-Clause for details.
//  In the following license statements, "software" refers to the
//  "source code" of the complete hardware/software system.
//
//  Copyright 2019,
//                    Embedded Intelligent Systems Lab (EISL)
//                    Deparment of Computer Science
//                    National Chiao Tung Uniersity
//                    Hsinchu, Taiwan.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
// =============================================================================
#pragma once
#include <stdint.h>

#define BLOCK_SIZE 512       // disk block size.
#define CLUSTER_SIZE 8192    // disk cluster size.
#define MAX_FAT_SIZE 512     // max FAT entry number, must be multiple of 128
#define FAT_BUF_LBA_SIZE (MAX_FAT_SIZE*sizeof(uint32_t))/BLOCK_SIZE

// -----------------------------------------------------------------------------
//    MBR Data Structures
// -----------------------------------------------------------------------------
typedef struct __attribute__ ((packed)) chs
{
    uint8_t cylinder;
    uint8_t head;
    uint8_t sector;
} chs;

typedef struct __attribute__ ((packed)) mbr
{
    uint8_t code[440];
    uint32_t disk_signature;
    uint16_t unused;
    struct mbr_partition
    {
        uint8_t status;
        struct chs first_sector;
        uint8_t partition_type;
        struct chs last_sector;
        uint32_t first_sector_lba;
        uint32_t sectors;
    } partition[4];
    uint8_t mbr_signature[2];
} mbr;

// -----------------------------------------------------------------------------
//    GBT Data Structures for Partition Tables
// -----------------------------------------------------------------------------
// LBA 0: Protective MBR
// ignored here

// Partition Table Header (LBA 1)
typedef struct gpt_pth
{
    uint32_t signature_low;
    uint32_t signature_high;
    uint32_t revision;
    uint32_t header_size;       //! little endian, usually 0x5c = 92
    uint32_t crc_header;
    uint32_t reserved;          //! must be 0
    uint64_t current_lba;
    uint64_t backup_lba;
    uint64_t first_usable_lba;
    uint64_t last_usable_lba;
    uint8_t disk_guid[16];
    uint64_t partition_entries_lba;
    uint32_t nr_partition_entries;
    uint32_t size_partition_entry;  //! usually 0x80 = 128
    uint32_t crc_partition_entry;
} gpt_pth_t;

// Partition Entries (LBA 2-33)
typedef struct partition_entries
{
    uint8_t partition_type_guid[16];
    uint8_t partition_guid[16];
    uint64_t first_lba;
    uint64_t last_lba;          //! inclusive
    uint64_t attributes;
    uint8_t name[72];           //! utf16 encoded
} partition_entries_t;


// -----------------------------------------------------------------------------
//    FAT32 Data Structures
// -----------------------------------------------------------------------------
typedef struct __attribute__ ((packed))
{
    uint8_t BS_JmpBoot[3];
    uint8_t BS_OEMName[8];
    uint16_t BPB_BytsPerSec;
    uint8_t BPB_SecPerClus;
    uint16_t BPB_RsvdSecCnt;
    uint8_t BPB_NumFATs;
    uint16_t BPB_RootEntCnt;
    uint16_t BPB_TotSec16;
    uint8_t BPB_Media;
    uint16_t BPB_FATSz16;
    uint16_t BPB_SecPerTrk;
    uint16_t BPB_NumHeads;
    uint32_t BPB_HiddSec;
    uint32_t BPB_TotSec32;

    uint32_t BPB_FATSz32;
    uint16_t BPB_ExtFlags;
    uint16_t BPB_FSVer;
    uint32_t BPB_RootClus;
    uint16_t BPB_FSInfo;
    uint16_t BPB_BkBootSec;
    uint8_t BPB_Reserved[12];
    uint8_t BS_DrvNum;
    uint8_t BS_Reserved1;
    uint8_t BS_BootSig;
    uint8_t BS_VolID[4];
    uint8_t BS_VolLab[11];
    uint8_t BS_FilSysType[8];
} boot_sector;

typedef struct __attribute__ ((packed))
{
    uint8_t DIR_Name[11];
    uint8_t DIR_Attr;
    uint8_t DIR_NTRes;
    uint8_t DIR_CrtTimeTenth;
    uint16_t DIR_CrtTime;
    uint16_t DIR_CrtDate;
    uint16_t DIR_LstAccDate;
    uint16_t DIR_FstClusHI;
    uint16_t DIR_WrtTime;
    uint16_t DIR_WrtDate;
    uint16_t DIR_FstClusLO;
    uint32_t DIR_FileSize;
} dir_entry;

/*
 * Function: copy_file
 * ----------------------------
 *   copy specific file from sd to dst
 *
 *   dst: destination 
 *   bs: fat32 boot sector
 *   first_lba: first lba of fat32 partition
 *   entry: source file entry
 *
 *   returns: zero if success else non-zero int
 */
int copy_file(uint8_t *dst, boot_sector *bs, uint32_t first_lba, dir_entry *entry);

/* test for sd_write */
// int write_file(boot_sector *bs, uint32_t first_lba, dir_entry *entry, char *lba_buf);

/*
 * Function: get_next_cluster
 * ----------------------------
 *   dynamic copy fat table from sd card and return next cluster number
 *
 *   target: current cluster num
 *   fat_buf: fat buffer
 *   buf_base: fat buffer start
 *   fat_start_lba: fat start lba
 *
 *   returns: next cluster number of target
 */
uint32_t get_next_cluster(uint32_t target, int *buf_base, uint32_t fat_start_lba);

/*
 * Function: get_partintion_first_lba
 * ----------------------------
 *   find nth partition start lba from MBR/GPT
 *
 *   partition_no: number of partition to find
 *
 *   returns: first lab of partition if success else zero
 */
uint64_t get_partition_first_lba(uint32_t partition_no);

/*
 * Function: read_file
 * ----------------------------
 *   Rreads a root directory file from the SD card to memory.
 *
 *   fname: input file name
 *
 *   fdata: memory address to load the data of the entire file
 *
 *   returns: zero if success, -1 if error occurred.
 */
uint32_t read_file(char *fname, uint8_t *fdata);

