module	t_Lab2_half_add();
	wire	sum, cout;
	reg		a, b;
	
	//instantiate device under test
	Lab2_half_add	M1(a, b, sum, cout);
	//apply inputs one at a time
	initial	begin
		a = 1'b0; b = 1'b0;
		#50;
        a = 1'b0; b = 1'b1;
        #50;
        a = 1'b1; b = 1'b0;
        #50;
        a = 1'b1; b = 1'b1;
	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_half_add.vcd");
		$dumpvars;
	end
endmodule