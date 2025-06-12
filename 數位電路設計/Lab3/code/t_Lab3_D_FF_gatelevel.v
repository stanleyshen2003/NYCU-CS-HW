module t_Lab3_D_Latch_gatelevel();

    reg D, Clk;
    wire Q, Qb ;
    Lab3_D_FF_gatelevel FF(D, Clk, Q, Qb);
    initial begin
        Clk = 0;
        forever #5 Clk = ~Clk;
    end
    initial fork
        #0 D = 0;
        #7 D = 1;
        #17 D = 0;
        #37 D = 1;
        #47 D = 0;
        #57 D = 1;
        #77 D = 0;
        #81 D = 1;
        #100 $finish;
    join
    initial begin
		$dumpfile("t_Lab3_D_FF_gatelevel.vcd");
		$dumpvars;
	end



endmodule