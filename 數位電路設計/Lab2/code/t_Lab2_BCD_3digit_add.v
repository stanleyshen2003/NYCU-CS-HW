module t_Lab2_BCD_3digit_add();
	wire [11:0]   S1;
    wire  cout1;
	reg [11:0]    A, B;
    reg  cin;
	
	Lab2_BCD_3digit_add	BCD_3digit(A, B, cin, S1, cout1);
    
	initial	begin
        A = 'h000; B = 'h000; cin = 1'b1;
        #25;
        A = 'h999; B = 'h999; cin = 1'b1;
        #25;
        A = 'h682; B = 'h835; cin = 1'b0;
        #25;
        A = 'h451; B = 'h069; cin = 1'b0;
        #25;
        A = 'h387; B = 'h616; cin = 1'b1;
        #25;
        A = 'h765; B = 'h943; cin = 1'b0;
        #25;
        A = 'h585; B = 'h556; cin = 1'b0;
        #25;
        A = 'h948; B = 'h051; cin = 1'b1;
        #25;

	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_BCD_3digit_add.vcd");
		$dumpvars;
	end


endmodule