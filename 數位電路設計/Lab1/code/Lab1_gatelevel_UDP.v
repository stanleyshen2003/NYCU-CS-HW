module	Lab1_gatelevel_UDP(F, A, B, C, D);
	output	F;
	input	A, B, C, D;
	wire	B_invert,D_invert,nor_noD,nor_D,or_AC;
	
	not	not1(B_invert, B);
	Lab1_UDP udp1(.F(nor_noD), .A(A), .B(B_invert), .C(C));

	not	not2(D_invert, D);
	or	or1(or_AC, A, C);
    Lab1_UDP udp2(.F(nor_D), .A(or_AC), .B(B), .C(D_invert));
	or	or2(F, nor_D, nor_noD);
endmodule