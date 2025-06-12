module t_Lab2_BCD_3digit_add_sub();
	wire [11:0]   BCD_R;
    wire  kout;
	reg [11:0]    BCD_X, BCD_Y;
    reg  mode;
	
	Lab2_BCD_3digit_add_sub	BCD_3digit(BCD_X, BCD_Y, mode, BCD_R, kout);
    
	initial	begin
        BCD_X = 'h000; BCD_Y = 'h000; mode = 1'b0;
        #25;
        BCD_X = 'h999; BCD_Y = 'h999; mode = 1'b0;
        #25;
        BCD_X = 'h548; BCD_Y = 'h459; mode = 1'b0;
        #25;
        BCD_X = 'h999; BCD_Y = 'h999; mode = 1'b1;
        #25;
        BCD_X = 'h569; BCD_Y = 'h568; mode = 1'b1;
        #25;
        BCD_X = 'h108; BCD_Y = 'h051; mode = 1'b1;
        #25;
        BCD_X = 'h387; BCD_Y = 'h616; mode = 1'b1;
        #25;
        BCD_X = 'h765; BCD_Y = 'h943; mode = 1'b1;
        #25;

	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_BCD_3digit_add_sub.vcd");
		$dumpvars;
	end


endmodule