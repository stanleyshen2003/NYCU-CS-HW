// =============================================================================
//  Program : fat32.c
//  Author  : ChihYu Hsiang 
//  Date    : Sep/25/2019
// -----------------------------------------------------------------------------
//  Description:
//  This is fat32 tool body for aquila bootrom that can parse
//  fat32 header and copy elf and binary code to memory.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  June/01/2020, by Jen-Yu Chi:
//    Added a do-loop for search all entries of the root directory.
//
//  Nov/22/2021, by Chun-Jen Tsai:
//    Added the read_file() and long2short() function.
//
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sd.h"
#include "fat32.h"

#define PARTITION_NO 0          // Always read from the first partition.

// Declare some private buffers.
static uint32_t fat[MAX_FAT_SIZE];  // for get_next_cluster()
static uint8_t lba_buf[BLOCK_SIZE]; // for copy_file() and read_file()

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
 *   returns: zero if fail else the file size in bytes.
 */
int copy_file(uint8_t *dst, boot_sector *bs, uint32_t first_lba, dir_entry *entry)
{
    int LBAPerSec = bs->BPB_BytsPerSec / BLOCK_SIZE;
    int LBAPerClus = bs->BPB_SecPerClus * LBAPerSec;
    uint16_t fat_start_sector = bs->BPB_RsvdSecCnt;
    uint32_t fat_sectors = bs->BPB_FATSz32 * bs->BPB_NumFATs;
    uint32_t data_start_lba = first_lba + (fat_start_sector+fat_sectors)*LBAPerSec;
    uint32_t cnt = 0;
    int buf_base = -1;
    int fat_base = first_lba + fat_start_sector * LBAPerSec;
    uint32_t cluster_num = entry->DIR_FstClusHI << 16;

    cluster_num += entry->DIR_FstClusLO;
    //printf("Loading the file, size = 0x%x, wait ...\n", entry->DIR_FileSize);

    while (cluster_num < 0x0FFFFFF8)
    {
        int res = sd_copy(dst, data_start_lba+(cluster_num-2)*LBAPerClus, LBAPerClus);
        if (res != 0)
        {
            printf("SD card failed at 170!\n");
            printf("sd copy return value: %d\n", res);
            return 0;
        }
        cnt++;
        dst += bs->BPB_SecPerClus*bs->BPB_BytsPerSec;
        cluster_num = get_next_cluster(cluster_num, &buf_base, fat_base);
    }
    // printf("Total clusters loaded: %d.\n", cnt);
    return entry->DIR_FileSize;
}

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
uint32_t get_next_cluster(uint32_t target, int *buf_base, uint32_t fat_start_lba)
{
    uint32_t base = target / MAX_FAT_SIZE;
    uint32_t offset = target % MAX_FAT_SIZE;

    if (*buf_base == -1 || base != *buf_base)
    {
        sd_copy(fat, fat_start_lba+base*FAT_BUF_LBA_SIZE, FAT_BUF_LBA_SIZE);
        *buf_base = base;
    }
    return fat[offset];
}

/*
 * Function: get_partition_first_lba
 * ----------------------------
 *   find nth partition start lba from MBR/GPT
 *
 *   part_no: the partition number to find
 *
 *   returns: first lab of partition if success else zero
 */
uint64_t get_partition_first_lba(uint32_t part_no)
{
    int res;
    uint64_t first_lba;

    if ((res = init_sd()) != 0)
    {
        printf("Can't initialize the SD card ... exiting %d.\n", res);
        return 0;
    }

    //printf("SD card initialized!\n");

    // At this point, we do not know if the disk is partitioned using
    // or GBT format. So we assume MBR first, then try GBT.

    // ============================
    //       Try to read MBR
    // ============================
    res = sd_copy(lba_buf, 0, 1);   //lba0
    if (res != 0)
    {
        printf("SD card failed with returned value: %d\n", res);
        return 0;
    }
    mbr *lba0 = (mbr *) lba_buf;
    first_lba = 0;
    if (lba0->partition[part_no].partition_type != 0xEE)
    {
        if (part_no > 3)
        {
            printf("\nMBR partition number should be < 4\n");
            return 0;
        }
        else
        {
            first_lba = lba0->partition[part_no].first_sector_lba;
        }
    }

    if (first_lba == 0) // Ok, now we know the disk is not partitioned using MBR format.
    {
		// ============================
		//       Try to read GPT
		// ============================
        res = sd_copy(lba_buf, 1, 1);   //lba1

        if (res != 0)
        {
            printf("SD card failed!\n");
            printf("sd copy return value: %d\n", res);
            return 0;
        }

        gpt_pth_t *lba1 = (gpt_pth_t *) lba_buf;
        uint32_t partition_entries_lba = (uint32_t) lba1->partition_entries_lba;
        res = sd_copy(lba_buf, partition_entries_lba, 1);

        if (res != 0)
        {
            printf("SD card failed!\n");
            printf("sd copy return value: %d\n", res);
            return 0;
        }

        partition_entries_t *boot = (partition_entries_t *) (lba_buf+(part_no*128));
        first_lba = boot->first_lba;
    }
    return first_lba;
}

#define TO_UPPER(x) ((x)>='a'?(x)<='z'?(x)-('a'-'A'):(x):(x))

/*
 * Function: long2short
 * ----------------------------
 *   convert a FAT32 long file name to the old FAT 8+3 file name.
 *
 *   fname: input file name
 *
 *   fname83: output file name
 *
 *   returns: none.
 */
void long2short(char *fname, char *fname83)
{
    char *name, *suffix;
    int idx, jdx, len, has_dot;

    for (idx = 0; idx < 11; idx++) fname83[idx] = 0x20; // space
    name = fname;
    suffix = fname;
    idx = has_dot = 0;
    while (fname[idx] != '.' && fname[idx] != 0) suffix++, idx++;
    if (fname[idx] == '.') suffix++, has_dot = 1;
    len = (int) suffix - (int) name - has_dot;

    for (idx = 0; idx < ((len > 8)? 8 : len); idx++)
    {
        fname83[idx] = TO_UPPER(fname[idx]);
    }
    if (*suffix != 0) // add suffix
    {
        for (jdx = 0; jdx < 3; jdx++)
        {
            if (suffix[jdx] != 0)
            {
                fname83[8 + jdx] = TO_UPPER(suffix[jdx]);
            }
            else break;
        }
    }

    // Change a long filename to a short one.
    // Note that this function always assume "~1" to 8+3 converted
    // names.  So, it does not work in all cases.
    if (len > 8)
    {
        fname83[6] = '~';
        fname83[7] = '1';
    }
}

/*
 * Function: read_file
 * ----------------------------
 *   Rreads a root directory file from the SD card to memory.
 *
 *   fname: input file name
 *
 *   fdata: memory address to load the data of the entire file
 *
 *   returns: file size if success, 0 if error occurred.
 */
uint32_t read_file(char *fname, uint8_t *fdata)
{
    uint32_t first_lba;
    uint16_t fat_start_sector;
    uint32_t fat_sectors;
    uint32_t data_start_sector;
    int LBAPerSec, LBAPerClus, BytesPerClus;
    uint8_t *clus_buf;
    boot_sector *bs;
    dir_entry *entries, *file_entry;
    uint32_t cluster_num;
    int fat_base, dat_base, buf_base;
    char filename[11];
    uint32_t idx, size;

    // File name conversion to 8+3 format.
    long2short(fname, filename);
    first_lba = get_partition_first_lba(PARTITION_NO);
    if (first_lba == 0) return 0;

    // Parse the boot section.
    sd_copy(lba_buf, first_lba, 1);
    bs = (boot_sector *) lba_buf;

    fat_start_sector = bs->BPB_RsvdSecCnt;
    fat_sectors = bs->BPB_FATSz32 * bs->BPB_NumFATs;
    data_start_sector = fat_start_sector + fat_sectors;
    LBAPerSec = bs->BPB_BytsPerSec / BLOCK_SIZE;
    LBAPerClus = bs->BPB_SecPerClus * LBAPerSec;
    BytesPerClus = bs->BPB_SecPerClus * bs->BPB_BytsPerSec;

    clus_buf = (uint8_t *) malloc(BytesPerClus);
    if (clus_buf == NULL)
    {
        printf("read_file: out of memory.\n");
        return 0;
    }

    fat_base = first_lba + fat_start_sector * LBAPerSec;
    dat_base = first_lba + data_start_sector * LBAPerSec;
    buf_base = -1;
    cluster_num = 2;
    size = 0;
    do
    {
        // Read the root directory block from the SD card.
        sd_copy(clus_buf, dat_base + (cluster_num-2)*LBAPerClus, LBAPerClus);
        entries = (dir_entry *) clus_buf;
        for (idx = 0; idx < BytesPerClus/32; idx++)
        {
            file_entry = entries + idx;
            if (file_entry->DIR_Name[0] != 0xE5 && file_entry->DIR_Attr != 0x0F)
            {
                if (!strncmp((char *) file_entry->DIR_Name, filename, 11))
                {
                    // Found the file, now start loading.
                    size = copy_file(fdata, bs, first_lba, file_entry);
                    break;
                }
            }
        }
        if (idx == BytesPerClus/32) break;
        cluster_num = get_next_cluster(cluster_num, &buf_base, fat_base);
    } while (cluster_num < 0x0FFFFFF8);

    free(clus_buf);
    return size;
}
