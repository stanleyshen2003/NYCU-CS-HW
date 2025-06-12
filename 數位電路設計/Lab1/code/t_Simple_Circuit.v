module	t_Simple_Circuit();
	wire	D1, E1, D2, E2;
	reg		A, B, C;
	
	//instantiate device under test
	Simple_Circuit	M1(A, B, C, D1, E1);
	Simple_Circuit_prop_delay	M2(A, B, C, D2, E2);
	
	//apply inputs one at a time
	initial	begin
		A=1'b0; B=1'b0; C=1'b0;
		#100 A=1'b1; B=1'b1; C=1'b1; 
	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab1_Simple_Circuit.vcd");
		$dumpvars;
	end
endmodule