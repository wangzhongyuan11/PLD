`timescale 1ns/1ps
module tb_feature_banks;
  logic clk=0;always #4 clk=~clk;
  logic resetn=0;
  logic [1:0] page_locked=0;
  logic w_page=0,w_bank=0,w_valid=0,w_ready;
  logic [5:0] w_addr=0,r_even_addr=0,r_odd_addr=0;
  logic [63:0] w_data=0;
  logic r_page=0,r_swap=0,r_valid=0,r_ready;
  logic [127:0] m_data;
  logic m_valid,m_ready=0;
  feature_banks #(.AW(6)) dut(.*);
  logic [63:0] reference_mem [0:1][0:1][0:63];
  logic [127:0] expected [0:4095];
  logic [127:0] held_data;
  logic held=0;
  integer wr=0,rd=0,checked=0,writes=0,rejected=0,seed=4128,cycles=0;
  always @(posedge clk) begin
    cycles=cycles+1;
    if (!resetn) begin wr=0;rd=0;held=0;end
    else begin
      if (w_ready && page_locked[w_page]) $fatal(1,"write admitted to locked page");
      if (held && (!m_valid || m_data!==held_data)) $fatal(1,"stalled read changed");
      held=m_valid && !m_ready;held_data=m_data;
      if (m_valid && m_ready) begin
        if (rd>=wr || m_data!==expected[rd]) $fatal(1,"bank read mismatch %0d",rd);
        rd=rd+1;checked=checked+1;
      end
      if (r_valid && r_ready) begin
        if (r_swap) expected[wr]={reference_mem[r_page][0][r_even_addr],reference_mem[r_page][1][r_odd_addr]};
        else expected[wr]={reference_mem[r_page][1][r_odd_addr],reference_mem[r_page][0][r_even_addr]};
        wr=wr+1;
      end
      if (w_valid && w_ready) begin
        reference_mem[w_page][w_bank][w_addr]=w_data;writes=writes+1;
      end
      if (w_valid && !w_ready) rejected=rejected+1;
    end
    if (cycles>10000) $fatal(1,"timeout");
  end
  initial begin
    repeat(3) @(negedge clk);resetn=1;
    for (int p=0;p<2;p++)
      for (int b=0;b<2;b++)
        for (int a=0;a<64;a++) begin
          @(negedge clk);w_valid=1;w_page=p;w_bank=b;w_addr=a;
          w_data={$random(seed),$random(seed)};
        end
    @(negedge clk);w_valid=0;
    // Simultaneous load into inactive page and arbitrary paired reads from active.
    for (int phase=0;phase<4;phase++) begin
      page_locked=phase%2 ? 2'b10 : 2'b01;
      for (int n=0;n<500;n++) begin
        @(negedge clk);
        r_page=phase%2;r_swap=$random(seed);r_even_addr=$random(seed);r_odd_addr=$random(seed);
        r_valid=1;m_ready=$unsigned($random(seed))%4!=0;
        w_page=(n%13==0) ? r_page : !r_page;
        w_bank=$random(seed);w_addr=$random(seed);w_data={$random(seed),$random(seed)};w_valid=1;
      end
      @(negedge clk);r_valid=0;w_valid=0;m_ready=1;
      repeat(3) @(negedge clk);
      if (wr!=rd) $fatal(1,"missing read result");
    end
    resetn=0;repeat(3) @(negedge clk);resetn=1;
    @(negedge clk);r_valid=1;m_ready=1;r_page=0;r_even_addr=3;r_odd_addr=9;
    @(negedge clk);r_valid=0;
    repeat(3) @(negedge clk);
    if (wr!=rd || checked<1000 || rejected==0) $fatal(1,"incomplete coverage");
    $display("PASS feature_banks checked=%0d writes=%0d rejected_locked=%0d",checked,writes,rejected);
    $finish;
  end
endmodule
