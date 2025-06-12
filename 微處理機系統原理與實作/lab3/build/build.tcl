#*****************************************************************************************
#  Create the aquial_bare workspace.
#*****************************************************************************************

# Adjust verison differences of the script. There is a change to the directory
# structure since Vivado 2021.1
#
set ver [version -short]
scan $ver %d.%d vmajor vminor
if { $vmajor > 2020 } {
    set gen_srcs gen
} else {
    set gen_srcs srcs
}

# Set the reference directory for source file relative paths
set origin_dir "."

# Set the project name
set proj_name "aquila_analysis"

# Create project
create_project ${proj_name} ${origin_dir}/${proj_name} -part xc7a100tcsg324-1

# Set project properties
set obj [current_project]
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "enable_resource_estimation" -value "0" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "simulator_language" -value "Mixed" -objects $obj
set_property -name "xpm_libraries" -value "XPM_CDC XPM_MEMORY" -objects $obj

# Create 'sources_1' fileset (if not found)
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

# Set 'sources_1' fileset object
# Import local design files
set files [list \
 [file normalize "$origin_dir/src/core_rtl/alu.v" ]\
 [file normalize "$origin_dir/src/core_rtl/aquila_config.vh" ]\
 [file normalize "$origin_dir/src/core_rtl/aquila_top.v" ]\
 [file normalize "$origin_dir/src/core_rtl/atomic_unit.v" ]\
 [file normalize "$origin_dir/src/core_rtl/bcu.v" ]\
 [file normalize "$origin_dir/src/core_rtl/bpu.v" ]\
 [file normalize "$origin_dir/src/core_rtl/clint.v" ]\
 [file normalize "$origin_dir/src/core_rtl/core_top.v" ]\
 [file normalize "$origin_dir/src/core_rtl/csr_file.v" ]\
 [file normalize "$origin_dir/src/core_rtl/dcache.v" ]\
 [file normalize "$origin_dir/src/core_rtl/decode.v" ]\
 [file normalize "$origin_dir/src/core_rtl/distri_ram.v" ]\
 [file normalize "$origin_dir/src/core_rtl/execute.v" ]\
 [file normalize "$origin_dir/src/core_rtl/fetch.v" ]\
 [file normalize "$origin_dir/src/core_rtl/forwarding_unit.v" ]\
 [file normalize "$origin_dir/src/core_rtl/icache.v" ]\
 [file normalize "$origin_dir/src/core_rtl/memory.v" ]\
 [file normalize "$origin_dir/src/core_rtl/muldiv.v" ]\
 [file normalize "$origin_dir/src/core_rtl/pipeline_control.v" ]\
 [file normalize "$origin_dir/src/core_rtl/program_counter.v" ]\
 [file normalize "$origin_dir/src/core_rtl/reg_file.v" ]\
 [file normalize "$origin_dir/src/core_rtl/sram.v" ]\
 [file normalize "$origin_dir/src/core_rtl/sram_dp.v" ]\
 [file normalize "$origin_dir/src/core_rtl/writeback.v" ]\
 [file normalize "$origin_dir/src/soc_rtl/uart.v" ]\
 [file normalize "$origin_dir/src/soc_rtl/soc_top.v" ]\
 [file normalize "$origin_dir/src/soc_rtl/core2axi_if.v"]\
 [file normalize "$origin_dir/src/soc_rtl/cdc_sync.v" ]\
 [file normalize "$origin_dir/src/soc_rtl/mem_arbiter.v" ]\
 [file normalize "$origin_dir/src/mem/uartboot.mem" ]\
]
set imported_files [import_files -fileset sources_1 $files]

# Define the target board to "ARTY" for synthesis
set_property verilog_define ARTY=true [get_filesets sources_1]

# Set 'sources_1' fileset properties
set obj [get_filesets sources_1]
set_property -name "top" -value "soc_top" -objects $obj
set files "*.mem"
set file_obj [get_files -of_objects [get_filesets sources_1] [list "$files"]]
set_property -name "file_type" -value "Memory File" -objects $file_obj

# Create 'constrs_1' fileset (if not found)
if {[string equal [get_filesets -quiet constrs_1] ""]} {
  create_fileset -constrset constrs_1
}

# Set 'constrs_1' fileset properties for local files
# Import constraint files and set file properties
set files [list \
  [file normalize "$origin_dir/src/xdc/arty.xdc"]\
]
set imported_files [import_files -fileset constrs_1 $files]
set files "*.xdc"
set file_objs [get_files -of_objects [get_filesets constrs_1] [list "$files"]]
set_property -name "file_type" -value "XDC" -objects $file_objs

# Create 'sim_1' fileset (if not found)
if {[string equal [get_filesets -quiet sim_1] ""]} {
  create_fileset -srcset sim_1
}

# Set 'sim_1' fileset object
# Import local simulation files
set files [list \
 [file normalize "$origin_dir/src/soc_rtl/soc_tb.v" ]\
 [file normalize "$origin_dir/src/soc_rtl/mig_7series_sim.v" ]\
]
set imported_files [import_files -fileset sim_1 $files]

# Define the target board to "ARTY" for simulation
set_property verilog_define ARTY=true [get_filesets sim_1]

set obj [get_filesets sim_1]
set_property -name "top" -value "soc_tb" -objects $obj
set_property -name "top_lib" -value "xil_defaultlib" -objects $obj

# Adding an Asynchronous FIFO Addr IP
create_ip -name fifo_generator -vendor xilinx.com -library ip -module_name async_fifo_addr
set_property -dict [list \
    CONFIG.fifo_implementation {independent_clocks_distributed_ram} \
    CONFIG.synchronization_stages {3} \
    CONFIG.input_data_width {32} \
    CONFIG.input_depth {16} \
    CONFIG.reset_pin {false} \
    CONFIG.performance_options {first_word_fall_through} ] [get_ips async_fifo_addr]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/async_fifo_addr/async_fifo_addr.xci]

# Adding an Asynchronous FIFO Data IP
create_ip -name fifo_generator -vendor xilinx.com -library ip -module_name async_fifo_data
set_property -dict [list \
    CONFIG.fifo_implementation {independent_clocks_distributed_ram} \
    CONFIG.synchronization_stages {3} \
    CONFIG.input_data_width {128} \
    CONFIG.input_depth {16} \
    CONFIG.reset_pin {false} \
    CONFIG.performance_options {first_word_fall_through} ] [get_ips async_fifo_data]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/async_fifo_data/async_fifo_data.xci]

# Adding an Asynchronous FIFO Signal IP
create_ip -name fifo_generator -vendor xilinx.com -library ip -module_name async_fifo_signal
set_property -dict [list \
    CONFIG.fifo_implementation {independent_clocks_distributed_ram} \
    CONFIG.synchronization_stages {3} \
    CONFIG.input_data_width {1} \
    CONFIG.input_depth {16} \
    CONFIG.reset_pin {false} \
    CONFIG.performance_options {first_word_fall_through} ] [get_ips async_fifo_signal]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/async_fifo_signal/async_fifo_signal.xci]

# Adding a Clock Wizard IP
# For some platforms, you may have to set CONFIG.PRIM_SOURCE {No_buffer}
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name clk_wiz_0
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {50.00000} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {166.66667} \
    CONFIG.CLKOUT3_USED {true} \
    CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {200.00000} \
    CONFIG.CLKOUT4_USED {false} \
    CONFIG.USE_RESET {false} \
    CONFIG.USE_LOCKED {false}] [get_ips clk_wiz_0]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/clk_wiz_0/clk_wiz_0.xci]

# Adding an SPI IP
create_ip -name axi_quad_spi -vendor xilinx.com -library ip -module_name axi_quad_spi_0
set_property -dict [ list \
   CONFIG.C_FIFO_DEPTH {256} \
   CONFIG.C_SCK_RATIO {4} \
   CONFIG.C_USE_STARTUP {0} \
   CONFIG.C_USE_STARTUP_INT {0} \
   CONFIG.QSPI_BOARD_INTERFACE {custom}] [get_ips axi_quad_spi_0]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/axi_quad_spi_0/axi_quad_spi_0.xci]
#update_compile_order -fileset sources_1
#set_property IS_ENABLED 0 [get_files ${proj_name}/${proj_name}.${gen_srcs}/sources_1/ip/axi_quad_spi_0/axi_quad_spi_0_board.xdc]

# Adding an MIG IP
create_ip -name mig_7series -vendor xilinx.com -library ip -module_name mig_7series_0
file copy $origin_dir/src/mig/mig-arty100t.prj ${proj_name}/${proj_name}.srcs/sources_1/ip/mig_7series_0/mig.prj
set_property -dict [list CONFIG.XML_INPUT_FILE {mig.prj}] [get_ips mig_7series_0]
generate_target {instantiation_template} [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/mig_7series_0/mig_7series_0.xci]
generate_target all [get_files ${proj_name}/${proj_name}.srcs/sources_1/ip/mig_7series_0/mig_7series_0.xci]

puts "INFO: Project created:${proj_name}"

