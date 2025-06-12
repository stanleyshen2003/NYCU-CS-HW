module Data_Memory(	clk_i, addr_i, data_i, MemRead_i, MemWrite_i, data_o);

// Interface
input				clk_i;
input	[31:0]		addr_i;
input	[31:0]		data_i;
input				MemRead_i;
input				MemWrite_i;
output	[31:0] 		data_o;

// Signals
reg		[31:0]		data_o;

// Memory
reg		[7:0]		Mem 			[0:127];	// address: 0x00~0x80
integer				i;

// For Testbench to debug
wire signed	[31:0]		memory			[0:31];
assign  memory[0] = {Mem[3], Mem[2], Mem[1], Mem[0]};
assign  memory[1] = {Mem[7], Mem[6], Mem[5], Mem[4]};
assign  memory[2] = {Mem[11], Mem[10], Mem[9], Mem[8]};
assign  memory[3] = {Mem[15], Mem[14], Mem[13], Mem[12]};
assign  memory[4] = {Mem[19], Mem[18], Mem[17], Mem[16]};
assign  memory[5] = {Mem[23], Mem[22], Mem[21], Mem[20]};
assign  memory[6] = {Mem[27], Mem[26], Mem[25], Mem[24]};
assign  memory[7] = {Mem[31], Mem[30], Mem[29], Mem[28]};
assign  memory[8] = {Mem[35], Mem[34], Mem[33], Mem[32]};
assign  memory[9] = {Mem[39], Mem[38], Mem[37], Mem[36]};
assign  memory[10] = {Mem[43], Mem[42], Mem[41], Mem[40]};
assign  memory[11] = {Mem[47], Mem[46], Mem[45], Mem[44]};
assign  memory[12] = {Mem[51], Mem[50], Mem[49], Mem[48]};
assign  memory[13] = {Mem[55], Mem[54], Mem[53], Mem[52]};
assign  memory[14] = {Mem[59], Mem[58], Mem[57], Mem[56]};
assign  memory[15] = {Mem[63], Mem[62], Mem[61], Mem[60]};
assign  memory[16] = {Mem[67], Mem[66], Mem[65], Mem[64]};
assign  memory[17] = {Mem[71], Mem[70], Mem[69], Mem[68]};
assign  memory[18] = {Mem[75], Mem[74], Mem[73], Mem[72]};
assign  memory[19] = {Mem[79], Mem[78], Mem[77], Mem[76]};
assign  memory[20] = {Mem[83], Mem[82], Mem[81], Mem[80]};
assign  memory[21] = {Mem[87], Mem[86], Mem[85], Mem[84]};
assign  memory[22] = {Mem[91], Mem[90], Mem[89], Mem[88]};
assign  memory[23] = {Mem[95], Mem[94], Mem[93], Mem[92]};
assign  memory[24] = {Mem[99], Mem[98], Mem[97], Mem[96]};
assign  memory[25] = {Mem[103], Mem[102], Mem[101], Mem[100]};
assign  memory[26] = {Mem[107], Mem[106], Mem[105], Mem[104]};
assign  memory[27] = {Mem[111], Mem[110], Mem[109], Mem[108]};
assign  memory[28] = {Mem[115], Mem[114], Mem[113], Mem[112]};
assign  memory[29] = {Mem[119], Mem[118], Mem[117], Mem[116]};
assign  memory[30] = {Mem[123], Mem[122], Mem[121], Mem[120]};
assign  memory[31] = {Mem[127], Mem[126], Mem[125], Mem[124]};

initial begin
	for(i=0; i<128; i=i+1)
		Mem[i] = 8'b0;
end 

always@(posedge clk_i) begin
    if(MemWrite_i) begin
		Mem[addr_i+3] <= data_i[31:24];
		Mem[addr_i+2] <= data_i[23:16];
		Mem[addr_i+1] <= data_i[15:8];
		Mem[addr_i]   <= data_i[7:0];
	end
end

always@(addr_i or MemRead_i) begin
	if(MemRead_i)
		data_o = {Mem[addr_i+3], Mem[addr_i+2], Mem[addr_i+1], Mem[addr_i]};
end

endmodule

