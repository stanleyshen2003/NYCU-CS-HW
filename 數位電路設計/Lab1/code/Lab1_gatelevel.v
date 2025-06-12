module	Lab1_gatelevel(F, A, B, C, D);
	output	F;
	input	A, B, C, D;
	wire	B_invert,D_invert,and_AB,and_ABC,nor_noD,nor_D,or_AC;
	
	not	not1(B_invert, B);
	and	and1(and_AB, A, B_invert);
	nor	nor1(nor_noD, and_AB, C);

	not	not2(D_invert, D);
	or	or1(or_AC, A, C);
	and	and2(and_ABC, B, or_AC);
	nor nor2(nor_D, and_ABC, D_invert);
	or	or2(F, nor_D, nor_noD);
endmodule