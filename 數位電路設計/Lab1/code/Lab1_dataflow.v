module	Lab1_dataflow(F, A, B, C, D);
	output	F;
	input	A, B, C, D;
	wire	B_invert,D_invert,and_AB,and_ABC,nor_noD,nor_D,or_AC;
	
	assign B_invert = ~B;
	assign and_AB = A & B_invert;
	assign nor_noD = ~(and_AB | C);

	assign D_invert = ~D;
	assign or_AC = A | C;
	assign and_ABC = B & or_AC;
	assign nor_D = ~(and_ABC | D_invert);
	assign F = nor_D | nor_noD;

endmodule