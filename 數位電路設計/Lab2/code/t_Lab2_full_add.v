module	t_Lab2_full_add();
	wire	sum, cout;
	reg		a, b, cin;
	
	Lab2_full_add	M1(a, b, cin, sum, cout);
    integer i;
    
	initial	begin
		for(i=0;i<8;i++) begin
            a = i[2];
            b = i[1];
            cin = i[0];
            #25;
        end

	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_full_add.vcd");
		$dumpvars;
	end
endmodule