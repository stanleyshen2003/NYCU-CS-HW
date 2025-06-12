module	t_Lab2_BCD_1digit_add();
	wire [3:0]   S1,S2;
    wire    cout1,cout2,cout3,cout;
	reg [3:0]    A, B;
    reg     cin;
	
	Lab2_BCD_1digit_add_df	df(A, B, cin, S1, cout1);
    Lab2_BCD_1digit_add_beh beh(A, B, cin, S2, cout2);
	initial	begin
        A = 4'b0000; B = 4'b0000; cin = 1'b1;
        #25;
        A = 4'b1001; B = 4'b1001; cin = 1'b1;
        #25;
        A = 4'b0011; B = 4'b0111; cin = 1'b0;
        #25;
        A = 4'b0101; B = 4'b1000; cin = 1'b0;
        #25;
        A = 4'b0011; B = 4'b0110; cin = 1'b1;
        #25;
        A = 4'b0110; B = 4'b0001; cin = 1'b1;
        #25;
        A = 4'b1001; B = 4'b0110; cin = 1'b0;
        #25;
        A = 4'b0101; B = 4'b0011; cin = 1'b0;
        #25;

	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_BCD_1digit_add.vcd");
		$dumpvars;
	end
endmodule