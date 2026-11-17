set io_skew   3.5
set host_skew 2.0
set rx_tsu    5.0
set rx_th     2.0

create_clock [get_ports clk]        -name clk     -period $clk_period
create_clock [get_ports {ui_in[0]}] -name sclk_in -period $sclk_period
set_clock_groups -asynchronous -group [get_clocks clk] -group [get_clocks sclk_in]

create_generated_clock -name sclk_out -source [get_ports clk] -divide_by 2 \
    [get_ports {uo_out[0]}]

set spi_in [get_ports {ui_in[1] ui_in[2] ui_in[4]}]
set_input_delay -max [expr $host_skew + $io_skew]    -clock sclk_in -clock_fall $spi_in
set_input_delay -min [expr -($host_skew + $io_skew)] -clock sclk_in -clock_fall $spi_in

set spi_out [get_ports {uo_out[1] uo_out[2] uo_out[3] uo_out[4]}]
set_output_delay -max [expr $rx_tsu + $io_skew]    -clock sclk_out $spi_out
set_output_delay -min [expr -($rx_th + $io_skew)]  -clock sclk_out $spi_out

set_multicycle_path -hold 1 -start -from [get_clocks clk] -to [get_clocks sclk_out]

set async_in [get_ports {ui_in[3] ui_in[5] ui_in[6] ui_in[7] uio_in[*] rst_n ena}]
set_input_delay 0 -clock clk $async_in
set_false_path -from $async_in
set static_out [get_ports {uo_out[5] uo_out[6] uo_out[7] uio_out[*] uio_oe[*]}]
set_output_delay 0 -clock clk $static_out
set_false_path -to $static_out

# --- Electrical, from base.sdc
set_max_fanout $::env(MAX_FANOUT_CONSTRAINT) [current_design]
if { [info exists ::env(MAX_TRANSITION_CONSTRAINT)] } {
    set_max_transition $::env(MAX_TRANSITION_CONSTRAINT) [current_design]
}
if { [info exists ::env(MAX_CAPACITANCE_CONSTRAINT)] } {
    set_max_capacitance $::env(MAX_CAPACITANCE_CONSTRAINT) [current_design]
}

if { ![info exists ::env(SYNTH_CLK_DRIVING_CELL)] } {
    set ::env(SYNTH_CLK_DRIVING_CELL) $::env(SYNTH_DRIVING_CELL)
}
set clk_ports   [get_ports {clk ui_in[0]}]
set data_inputs [get_ports {ui_in[1] ui_in[2] ui_in[3] ui_in[4] ui_in[5] ui_in[6] ui_in[7] uio_in[*] rst_n ena}]
set_driving_cell \
    -lib_cell [lindex [split $::env(SYNTH_DRIVING_CELL) "/"] 0] \
    -pin [lindex [split $::env(SYNTH_DRIVING_CELL) "/"] 1] \
    $data_inputs
set_driving_cell \
    -lib_cell [lindex [split $::env(SYNTH_CLK_DRIVING_CELL) "/"] 0] \
    -pin [lindex [split $::env(SYNTH_CLK_DRIVING_CELL) "/"] 1] \
    $clk_ports

# The tile port drives the TT mux, not the board
set_load [expr $::env(OUTPUT_CAP_LOAD) / 1000.0] [all_outputs]

set_clock_uncertainty $::env(CLOCK_UNCERTAINTY_CONSTRAINT) [all_clocks]
set_clock_transition  $::env(CLOCK_TRANSITION_CONSTRAINT)  [get_clocks {clk sclk_in}]

set_timing_derate -early [expr 1 - $::env(TIME_DERATING_CONSTRAINT) / 100.0]
set_timing_derate -late  [expr 1 + $::env(TIME_DERATING_CONSTRAINT) / 100.0]

if { [info exists ::env(OPENLANE_SDC_IDEAL_CLOCKS)] && $::env(OPENLANE_SDC_IDEAL_CLOCKS) } {
    unset_propagated_clock [all_clocks]
} else {
    set_propagated_clock [all_clocks]
}
