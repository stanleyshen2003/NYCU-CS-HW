module Reg_File( clk_i, rst_n, RSaddr_i, RTaddr_i, Wrtaddr_i, Wrtdata_i, RegWrite_i, RSdata_o, RTdata_o );
          
//I/O ports
input           clk_i;
input           rst_n;
input           RegWrite_i;
input  [5-1:0]  RSaddr_i;
input  [5-1:0]  RTaddr_i;
input  [5-1:0]  Wrtaddr_i;
input  [32-1:0] Wrtdata_i;

output [32-1:0] RSdata_o;
output [32-1:0] RTdata_o;   

//Internal signals/registers           
reg  signed [32-1:0] Reg_File [0:32-1];     //32 word registers
wire        [32-1:0] RSdata_o;
wire        [32-1:0] RTdata_o;

//Read the data
assign RSdata_o = Reg_File[RSaddr_i] ;
assign RTdata_o = Reg_File[RTaddr_i] ;   

//Writing data when postive edge clk_i and RegWrite_i was set.
always @( negedge rst_n or negedge clk_i  ) begin
    if(rst_n == 0) begin
	    Reg_File[0]  <= 0; Reg_File[1]  <= 0; Reg_File[2]  <= 0; Reg_File[3]  <= 0;
	    Reg_File[4]  <= 0; Reg_File[5]  <= 0; Reg_File[6]  <= 0; Reg_File[7]  <= 0;
        Reg_File[8]  <= 0; Reg_File[9]  <= 0; Reg_File[10] <= 0; Reg_File[11] <= 0;
	    Reg_File[12] <= 0; Reg_File[13] <= 0; Reg_File[14] <= 0; Reg_File[15] <= 0;
        Reg_File[16] <= 0; Reg_File[17] <= 0; Reg_File[18] <= 0; Reg_File[19] <= 0;      
        Reg_File[20] <= 0; Reg_File[21] <= 0; Reg_File[22] <= 0; Reg_File[23] <= 0;
        Reg_File[24] <= 0; Reg_File[25] <= 0; Reg_File[26] <= 0; Reg_File[27] <= 0;
        Reg_File[28] <= 0; Reg_File[29] <= 32'd128; Reg_File[30] <= 0; Reg_File[31] <= 0;
	end
    else begin
        if(RegWrite_i) 
            Reg_File[Wrtaddr_i] <= Wrtdata_i;	
		else 
		    Reg_File[Wrtaddr_i] <= Reg_File[Wrtaddr_i];
	end
end

endmodule     





                    
                    
