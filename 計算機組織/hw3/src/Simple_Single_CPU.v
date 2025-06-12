module Simple_Single_CPU( clk_i, rst_n );

//I/O port
input         clk_i;
input         rst_n;

//Internal Signles

wire [31:0] pcin, pcout, four, now_instruction, writeData, ALU_input1, ALU_input2, sign_extended, noUse, ALU_input2_chosen, result, shift_result;
assign four = 32'h00000004;
wire regDst,RegWrite, ALUSrc, noUse2, noUse3;
wire [2:0] ALUOP;
wire [3:0] ALU_operation;
wire [4:0] writeRg;
wire [1:0] FURslt;


//modules
Program_Counter PC(
        .clk_i(clk_i),      
	    .rst_n(rst_n),     
	    .pc_in_i(pcin) ,   
	    .pc_out_o(pcout) 
	    );
	
Adder Adder1(
        .src1_i(pcout),     
	    .src2_i(four),
	    .sum_o(pcin)    
	    );
	
Instr_Memory IM(
        .pc_addr_i(pcout),  
	    .instr_o(now_instruction)    
	    );

Mux2to1 #(.size(5)) Mux_Write_Reg(
        .data0_i(now_instruction[20:16]),
        .data1_i(now_instruction[15:11]),
        .select_i(RegDst),
        .data_o(writeRg)
        );	
		
Reg_File RF(
        .clk_i(clk_i),      
	    .rst_n(rst_n) ,     
        .RSaddr_i(now_instruction[25:21]) ,  
        .RTaddr_i(now_instruction[20:16]) ,  
        .RDaddr_i(writeRg) ,  
        .RDdata_i(writeData)  , 
        .RegWrite_i(RegWrite),
        .RSdata_o(ALU_input1) ,  
        .RTdata_o(ALU_input2)   
        );
	
Decoder Decoder(
        .instr_op_i(now_instruction[31:26]), 
	    .RegWrite_o(RegWrite), 
	    .ALUOp_o(ALUOP),   
	    .ALUSrc_o(ALUSrc),   
	    .RegDst_o(RegDst)   
		);

ALU_Ctrl AC(
        .funct_i(now_instruction[5:0]),   
        .ALUOp_i(ALUOP),   
        .ALU_operation_o(ALU_operation),
		.FURslt_o(FURslt)
        );
	
Sign_Extend SE(
        .data_i(now_instruction[15:0]),
        .data_o(sign_extended)
        );

Zero_Filled ZF(
        .data_i(now_instruction[15:0]),
        .data_o(noUse)
        );
		
Mux2to1 #(.size(32)) ALU_src2Src(
        .data0_i(ALU_input2),
        .data1_i(sign_extended),
        .select_i(ALUSrc),
        .data_o(ALU_input2_chosen)
        );	
		
ALU ALU(
		.aluSrc1(ALU_input1),
	    .aluSrc2(ALU_input2_chosen),
	    .ALU_operation_i(ALU_operation),
		.result(result),
		.zero(noUse2),
		.overflow(noUse3)
	    );
	    
wire leftright;
assign leftright = ~now_instruction[1];
		
Shifter shifter( 
		.result(shift_result), 
		.leftRight(leftright),
		.shamt(now_instruction[10:6]),
		.sftSrc(ALU_input2_chosen) 
		);
		
Mux3to1 #(.size(32)) RDdata_Source(
        .data0_i(result),
        .data1_i(shift_result),
		.data2_i(noUse),
        .select_i(FURslt),
        .data_o(writeData)
        );			

endmodule



