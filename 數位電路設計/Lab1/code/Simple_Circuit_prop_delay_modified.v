module	Simple_Circuit_prop_delay(A, B, C, D, E);
	output	D, E;
	input	A, B, C;
	wire	w1;
	
	or	#(20) G3(D, w1, E);
	and	#(30) G1(w1, A, B);
	not	#(10) G2(E, C);
	
endmodule