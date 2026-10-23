// Two ordered jobs, two independent input/parameter caches, one active compute.
// A cache entry is reusable only after its loader's checked DMA completion.
// Producer IDs + frame epoch prevent address reuse from becoming a false hit.
// Fault is terminal until a COMMON reset also quiesces the external DMA engines.
module tile_job_queue (
  input logic aclk,aresetn,clear_cache,
  input logic [511:0] s_descriptor,
  input logic s_valid,
  output logic s_ready,
  output logic [511:0] load_descriptor,run_descriptor,
  output logic load_input_slot,load_weight_slot,run_input_slot,run_weight_slot,
  output logic input_start,parameter_start,lut_start,run_start,
  input logic input_done,parameter_done,lut_done,core_done,writer_done,
  input logic [7:0] input_error,parameter_error,lut_error,writer_error,
  input logic [2:0] core_error,
  output logic [31:0] completion_tag,
  output logic completion_valid,
  input logic completion_ready,
  output logic busy,fault,
  output logic [15:0] error,
  output logic [31:0] error_tag,
  output logic [31:0] input_hits,parameter_hits,overlap_cycles
);
  typedef enum logic [2:0] {FREE,QUEUED,LOADING,READY,RUNNING,COMPLETE} qstate_t;
  typedef enum logic [1:0] {LOAD_IDLE,LOAD_TABLE,LOAD_DATA} lstate_t;
  qstate_t qs[2];lstate_t ls;
  logic [511:0] descriptor[2];
  logic [1:0] input_slot,weight_slot;
  logic wp,lp,rp,got_input,got_parameter,got_core,got_writer;
  logic [305:0] input_key[2],wanted_input;
  logic [72:0] parameter_key[2],wanted_parameter;
  logic [1:0] input_valid,parameter_valid,input_protected,parameter_protected;
  logic table_valid;logic [31:0] table_address;
  logic input_hit,parameter_hit,input_available,parameter_available;
  logic chosen_input,chosen_parameter,barrier_clear,legal;
  logic load_sent,table_sent;
  logic [511:0] pending;
  logic [7:0] flags,ci,co,split;
  logic [15:0] ih,iw,y,x,oh,ow;
  function automatic logic power2(input logic [15:0] v);
    return v!=0&&v<=256&&(v&(v-1))==0;
  endfunction
  function automatic logic [305:0] ikey(input logic [511:0] d);
    return {d[511:480],d[479:448],d[447:416],d[63:0],
            (d[378]?d[127:96]:32'd0),d[351:256],d[359:352],d[375:368],d[376],d[378]};
  endfunction
  function automatic logic [72:0] pkey(input logic [511:0] d);
    return {d[511:480],d[95:64],d[359:352],d[376]};
  endfunction
  assign pending=descriptor[lp];
  assign wanted_input=ikey(pending);assign wanted_parameter=pkey(pending);
  assign s_ready=aresetn&&!fault&&qs[wp]==FREE&&!clear_cache;
  assign completion_valid=aresetn&&!fault&&qs[rp]==COMPLETE;
  assign completion_tag=descriptor[rp][415:384];
  assign busy=qs[0]!=FREE||qs[1]!=FREE;
  assign load_descriptor=descriptor[lp];assign run_descriptor=descriptor[rp];
  assign load_input_slot=input_slot[lp];assign load_weight_slot=weight_slot[lp];
  assign run_input_slot=input_slot[rp];assign run_weight_slot=weight_slot[rp];
  always_comb begin
    input_protected=0;parameter_protected=0;
    for(int j=0;j<2;j++)if(qs[j]!=FREE&&qs[j]!=QUEUED)begin
      input_protected[input_slot[j]]=1;parameter_protected[weight_slot[j]]=1;
    end
    input_hit=0;parameter_hit=0;chosen_input=0;chosen_parameter=0;
    // Lowest matching slot wins deterministically, including an active read slot.
    for(int i=1;i>=0;i--)begin
      if(input_valid[i]&&input_key[i]==wanted_input)begin input_hit=1;chosen_input=1'(i);end
      if(parameter_valid[i]&&parameter_key[i]==wanted_parameter)begin parameter_hit=1;chosen_parameter=1'(i);end
    end
    input_available=input_hit||!(&input_protected);
    parameter_available=parameter_hit||!(&parameter_protected);
    if(!input_hit)chosen_input=input_protected[0];
    if(!parameter_hit)chosen_parameter=parameter_protected[0];
    // lp is ordered. The other queued job can be younger: never wait on it.
    barrier_clear=!pending[380]||(qs[!lp]==FREE||qs[!lp]==QUEUED);
    flags=s_descriptor[383:376];ci=s_descriptor[359:352];co=s_descriptor[367:360];split=s_descriptor[375:368];
    ih=s_descriptor[287:272];iw=s_descriptor[271:256];y=s_descriptor[319:304];x=s_descriptor[303:288];
    oh=s_descriptor[351:336];ow=s_descriptor[335:320];
    legal=power2(ih)&&power2(iw)&&oh!=0&&ow!=0&&oh<=ih&&ow<=iw&&
          {1'b0,y}+oh<=ih&&{1'b0,x}+ow<=iw&&ci!=0&&ci<=192&&co!=0&&co<=8&&flags[7:5]==0&&
          s_descriptor[2:0]==0&&s_descriptor[66:64]==0&&s_descriptor[130:128]==0&&s_descriptor[162:160]==0;
    if(flags[2])legal&=split!=0&&split<ci&&split[2:0]==0&&ci[2:0]==0&&
                              s_descriptor[34:32]==0&&s_descriptor[98:96]==0&&ih>=2&&iw>=2;
    else legal&=split==0;
    if(flags[3])legal&=oh[0]==0&&ow[0]==0&&y[0]==0&&x[0]==0&&
                              s_descriptor[194:192]==0&&s_descriptor[226:224]==0;
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin
      qs[0]<=FREE;qs[1]<=FREE;ls<=LOAD_IDLE;wp<=0;lp<=0;rp<=0;
      input_valid<=0;parameter_valid<=0;table_valid<=0;fault<=0;error<=0;error_tag<=0;
      input_start<=0;parameter_start<=0;lut_start<=0;run_start<=0;
      got_input<=0;got_parameter<=0;got_core<=0;got_writer<=0;
      input_hits<=0;parameter_hits<=0;overlap_cycles<=0;input_slot<=0;weight_slot<=0;
    end else begin
      input_start<=0;parameter_start<=0;lut_start<=0;run_start<=0;
      if(!fault)begin
        if(ls!=LOAD_IDLE&&qs[rp]==RUNNING)overlap_cycles<=overlap_cycles+1;
        if(clear_cache)begin
          if(busy)begin fault<=1;error<=16'h0102;error_tag<=completion_tag;end
          else begin input_valid<=0;parameter_valid<=0;table_valid<=0;end
        end
        if(s_valid&&s_ready)begin
          if(!legal)begin fault<=1;error<=16'h0101;error_tag<=s_descriptor[415:384];end
          else begin descriptor[wp]<=s_descriptor;qs[wp]<=QUEUED;wp<=!wp;end
        end
        case(ls)
          LOAD_IDLE:if(qs[lp]==QUEUED&&barrier_clear&&input_available&&parameter_available)begin
            input_slot[lp]<=chosen_input;weight_slot[lp]<=chosen_parameter;
            qs[lp]<=LOADING;got_input<=input_hit;got_parameter<=parameter_hit;
            if(input_hit)input_hits<=input_hits+1;
            else begin input_valid[chosen_input]<=0;input_key[chosen_input]<=wanted_input;end
            if(parameter_hit)parameter_hits<=parameter_hits+1;
            else begin parameter_valid[chosen_parameter]<=0;parameter_key[chosen_parameter]<=wanted_parameter;end
            // Start pulses occur only after assigned slots/descriptors are stable.
            ls<=LOAD_TABLE;
          end
          LOAD_TABLE:begin
            if(!got_input&&pending[378]&&(!table_valid||table_address!=pending[127:96]))begin
              if(!table_sent&&!lut_start&&!lut_done)lut_start<=1;
              if(lut_done)begin table_valid<=1;table_address<=pending[127:96];ls<=LOAD_DATA;end
            end else ls<=LOAD_DATA;
          end
          LOAD_DATA:begin
            // A sent flag is necessary: done can be many cycles after start.
            if(!load_sent)begin input_start<=!got_input;parameter_start<=!got_parameter;end
            if(input_done)begin got_input<=1;input_valid[load_input_slot]<=1;end
            if(parameter_done)begin got_parameter<=1;parameter_valid[load_weight_slot]<=1;end
            if(got_input&&got_parameter)begin qs[lp]<=READY;lp<=!lp;ls<=LOAD_IDLE;end
          end
        endcase
        if(qs[rp]==READY)begin
          run_start<=1;qs[rp]<=RUNNING;got_core<=0;got_writer<=0;
        end
        if(qs[rp]==RUNNING)begin
          if(core_done)got_core<=1;
          if(writer_done)got_writer<=1;
          if(got_core&&got_writer)qs[rp]<=COMPLETE;
        end
        if(completion_valid&&completion_ready)begin qs[rp]<=FREE;rp<=!rp;end
        if(input_error!=0||parameter_error!=0||lut_error!=0||core_error!=0||writer_error!=0)begin
          fault<=1;input_valid<=0;parameter_valid<=0;table_valid<=0;
          if(input_error!=0)begin error<={8'h02,input_error};error_tag<=pending[415:384];end
          else if(parameter_error!=0)begin error<={8'h03,parameter_error};error_tag<=pending[415:384];end
          else if(lut_error!=0)begin error<={8'h04,lut_error};error_tag<=pending[415:384];end
          else if(core_error!=0)begin error<={8'h05,5'd0,core_error};error_tag<=completion_tag;end
          else begin error<={8'h06,writer_error};error_tag<=completion_tag;end
        end
      end
    end
  end
  always_ff @(posedge aclk)begin
    if(!aresetn)begin load_sent<=0;table_sent<=0;end
    else begin
      if(ls!=LOAD_DATA)load_sent<=0;else load_sent<=1;
      if(ls!=LOAD_TABLE)table_sent<=0;else if(lut_start)table_sent<=1;
    end
  end
endmodule
