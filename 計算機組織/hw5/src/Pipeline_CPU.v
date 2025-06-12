module Pipeline_CPU( clk_i, rst_n );

//I/O port
input         clk_i;
input         rst_n;
`timescale 1ns / 1ps
//Internal Signles
wire [32-1:0] instr0, instr, PC_i, PC_o, ReadData1_temp, ReadData2_temp, WriteData, ReadData1, ReadData2;
wire [32-1:0] signextend_temp, ALUinput2, ALUResult, ALUResult_temp, ALUResult_temp2, ShifterResult, signextend;
wire [5-1:0] WriteReg_addr, Shifter_shamt;
wire [4-1:0] ALU_operation;
wire [3-1:0] ALUOP, ALUOP_temp;
wire [2-1:0] FURslt;
wire [2-1:0] RegDst, RegDst_temp, MemtoReg, MemtoReg_temp0, MemtoReg_temp1, MemtoReg_temp2;
wire RegWrite, RegWrite_temp0, RegWrite_temp1, RegWrite_temp2, ALUSrc, ALUSrc_temp, zero, zero_temp, overflow;
wire Jump, Branch, Branch_temp0, Branch_temp1, BranchType, MemWrite, MemWrite_temp0, MemWrite_temp1, MemRead, MemRead_temp0, MemRead_temp1;
wire [32-1:0] PC_add1,PC_add_temp1, PC_add2, PC_no_jump, PC_t, Mux3_result, DM_ReadData, PC_add2_temp, DM_ReadData_temp, PC_add_temp0;
wire Jr;
wire [9:0] instr_3;
assign Jr = ((instr[31:26] == 6'b000000) && (instr[20:0] == 21'd8)) ? 1 : 0;
/*
always@ (negedge clk_i) begin
    $display("instr : %b",instr);
    $display("RegDst : %b", RegDst);
    $display("ALUSrc : %b", ALUSrc);
    $display("ALUOP : %b", ALUOP); 
    $display("Branch : %b", Branch);
    $display("MemWrite : %b", MemWrite); 
    $display("MemRead : %b", MemRead);
    $display("MemtoReg : %b", MemtoReg);
    $display("RegWrite : %b\n", RegWrite);
    end
  */
//modules
Program_Counter PC(
    .clk_i(clk_i),
	.rst_n(rst_n),
	.pc_in_i(PC_i),
	.pc_out_o(PC_o)
	);

Adder Adder1(//next instruction
        .src1_i(PC_o), 
	.src2_i(32'd4),
	.sum_o(PC_add1)
	);

Pipeline_Reg #(.size(32)) reg11(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(PC_add1),
        .data_o(PC_add_temp0)
);

Pipeline_Reg #(.size(32)) reg21(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(PC_add_temp0),
        .data_o(PC_add_temp1)
);

/*always@(PC_o)
begin
    $display("pc_o: %b", PC_o);
	$display("pc+4: %b", PC_add1);
end
*/		
Adder Adder2(
        .src1_i(PC_add_temp1),
	.src2_i({signextend[29:0], 2'b00}),
	.sum_o(PC_add2_temp)
	);

Mux2to1 #(.size(32)) Mux_branch(
        .data0_i(PC_add1),
        .data1_i(PC_add2),
        .select_i(Branch & zero),    ////////////////
        .data_o(PC_i)
        );

Pipeline_Reg #(.size(32)) reg31(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(PC_add2_temp),
        .data_o(PC_add2)
);

/*always@(*)
begin
    $display("pc_add1: %b", PC_add1);
	$display("pc_add2: %b", PC_add2);
	$display("pc_no_jump: %b", PC_no_jump);
end*/

////////////////////////////////   first   ////////////////////////////////////     second    /////////////////////////

Instr_Memory IM(
        .pc_addr_i(PC_o),
	   .instr_o(instr0)
	   );

Pipeline_Reg #(.size(32)) reg12(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(instr0),
        .data_o(instr)
);
/*always@(*)
begin
    $display("PC_o: %b", PC_o);
	$display("instr: %b", instr);
end*/

///////////////////////////////////    last    //////////////////////////

Mux2to1 #(.size(5)) Mux_Write_Reg(
        .data0_i(instr_3[9:5]),
        .data1_i(instr_3[4:0]),
        .select_i(RegDst[0]),
        .data_o(WriteReg_addr)
);

Pipeline_Reg #(.size(10)) reg25(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(instr[20:11]),
        .data_o(instr_3)
);

wire [4:0] WriteReg_addr_temp0, WriteReg_addr_temp1;

Pipeline_Reg #(.size(5)) reg35(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(WriteReg_addr),
        .data_o(WriteReg_addr_temp0)
);

Pipeline_Reg #(.size(5)) reg43(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(WriteReg_addr_temp0),
        .data_o(WriteReg_addr_temp1)
);

/////////////////////////////    middle    ///////////////////////////

Reg_File RF(
        .clk_i(clk_i),
	    .rst_n(rst_n),
        .RSaddr_i(instr[25:21]),
        .RTaddr_i(instr[20:16]),
        .Wrtaddr_i(WriteReg_addr_temp1),
        .Wrtdata_i(WriteData),
        .RegWrite_i(RegWrite & (~Jr)),
        .RSdata_o(ReadData1_temp),
        .RTdata_o(ReadData2_temp)
        );

Pipeline_Reg #(.size(32)) reg22(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(ReadData1_temp),
        .data_o(ReadData1)
);
Pipeline_Reg #(.size(32)) reg23(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(ReadData2_temp),
        .data_o(ReadData2)
);


//////////////   decoder not done    ////////////
Decoder Control(
        .instr_op_i(instr[31:26]),
	.RegWrite_o(RegWrite_temp0),
	.ALUOp_o(ALUOP_temp),
	.ALUSrc_o(ALUSrc_temp),
	.RegDst_o(RegDst_temp),
	.Jump_o(Jump),
	.Branch_o(Branch_temp0),
	.BranchType_o(BranchType),
	.MemWrite_o(MemWrite_temp0),
	.MemRead_o(MemRead_temp0),
	.MemtoReg_o(MemtoReg_temp0)
	);

Pipeline_Reg #(.size(6)) regEX(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({RegDst_temp, ALUOP_temp, ALUSrc_temp}),
        .data_o({RegDst, ALUOP, ALUSrc})
);

Pipeline_Reg #(.size(3)) regMEM1(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({Branch_temp0, MemWrite_temp0, MemRead_temp0}),
        .data_o({Branch_temp1, MemWrite_temp1, MemRead_temp1})
);

Pipeline_Reg #(.size(3)) regMEM2(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({Branch_temp1, MemWrite_temp1, MemRead_temp1}),
        .data_o({Branch, MemWrite, MemRead})
);

Pipeline_Reg #(.size(3)) regWB1(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({MemtoReg_temp0, RegWrite_temp0}),
        .data_o({MemtoReg_temp1, RegWrite_temp1})
);

Pipeline_Reg #(.size(3)) regWB2(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({MemtoReg_temp1, RegWrite_temp1}),
        .data_o({MemtoReg_temp2, RegWrite_temp2})
);

Pipeline_Reg #(.size(3)) regWB3(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i({MemtoReg_temp2, RegWrite_temp2}),
        .data_o({MemtoReg, RegWrite})
);

///////////  AC not done   ///////////
ALU_Ctrl AC(
        .funct_i(signextend[5:0]),
        .ALUOp_i(ALUOP),
        .ALU_operation_o(ALU_operation),
	   .FURslt_o(FURslt)
        );

	
Sign_Extend SE(
        .data_i(instr[15:0]),
        .data_o(signextend_temp)
        );

Pipeline_Reg #(.size(32)) reg24(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(signextend_temp),
        .data_o(signextend)
);

/*always@(*)
begin
	$display("signextend: %d", signextend);
end*/

		
Mux2to1 #(.size(32)) ALU_src2Src(
        .data0_i(ReadData2),
        .data1_i(signextend),
        .select_i(ALUSrc),
        .data_o(ALUinput2)
        );	
/*
Mux2to1 #(.size(32)) Shifter_in( //srl sll sllv srlv
        .data0_i({27'd0,instr[10:6]}),//fill to 32 bit
        .data1_i(ReadData1),
        .select_i(ALU_operation[1]),
        .data_o(Shifter_shamt)
        ); // Shifter_shamt would cause warning(Mux output: 32b, shifter shamt: 5b)
*/
ALU ALU(
	.aluSrc1(ReadData1),
	.aluSrc2(ALUinput2),
	.ALU_operation_i(ALU_operation),
	.result(ALUResult_temp),
	.zero(zero_temp),
	.overflow(overflow)
	);

Pipeline_Reg #(.size(1)) reg32(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(zero_temp),
        .data_o(zero)
);

Pipeline_Reg #(.size(32)) reg33(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(ALUResult_temp),
        .data_o(ALUResult)
);

wire [31:0] ReadData2_temp2;

Pipeline_Reg #(.size(32)) reg34(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(ReadData2),
        .data_o(ReadData2_temp2)
);

/*always@(*)
begin
	$display("ReadData1: %d", ReadData1);
	$display("ALUinput2: %d", ALUinput2);
	$display("ALUResult: %d", ALUResult);
end*/


		
/*always@(*)
begin
	$display("ALUResult: %d", ALUResult);
	$display("FURslt: %d", FURslt);
	$display("Mux3_result: %d", Mux3_result);
end*/
Data_Memory DM(
	.clk_i(clk_i),
	.addr_i(ALUResult),
	.data_i(ReadData2_temp2),
	.MemRead_i(MemRead),
	.MemWrite_i(MemWrite),
	.data_o(DM_ReadData_temp)
	);

Pipeline_Reg #(.size(32)) reg41(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(DM_ReadData_temp),
        .data_o(DM_ReadData)
);

Pipeline_Reg #(.size(32)) reg42(
        .clk_i(clk_i),
        .rst_i(rst_n),
        .data_i(ALUResult),
        .data_o(ALUResult_temp2)
);
/*always@(*)
begin
	$display("Mux3_result: %d", $signed(Mux3_result));
	$display("ReadData2: %d", $signed(ReadData2));
end*/	
Mux2to1 #(.size(32)) Mux_Write( 
        .data0_i(ALUResult_temp2),
        .data1_i(DM_ReadData),
        .select_i(MemtoReg[0]),
        .data_o(WriteData)
        );
/*always@(*)
begin
	$display("WriteData: d", WriteData);
end*/
endmodule



