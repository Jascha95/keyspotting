import sys
sys.path.append('..')

from ipywidgets import interactive_output, Layout, IntSlider, Button, HBox, VBox, Label, Checkbox, FloatText, jslink

import dynapse1utils as ut
from dynapse1constants import *

bias_slider_style = {'description_width': 'initial'}
label_style = {'font_weight' : 'bold'}
grid_gap='10px 70px'

SYN_TAGS = ['ampa', 'nmda', 'gaba_a', 'gaba_b']
    
    
def make_interactive_bias_slider(model, chip_id, core_id, bias_tag):
    """ Creates an interactive iPyWidget slider for the given parameter.
    Requires the `model` as an argument  

    Args:
        model (Dynapse1model): device model handle
        chip_id (int): target chip 
        core_id (int): target core
        bias_tag (str): tag of the parameter to set

    Returns:
        slider_widget (HBox): Widget block containing the parameter label, the slider in 'index' format and the corresponding
                            value print in `tuple` format.
    """
    
    
    bias_name = BIAS_NAMES[BIAS_TAGS_LOOKUP[bias_tag]]
    bias_value = int(ut._get_bias(model, chip_id, core_id, bias_name, 'index'))
    bias_value_tuple = ut._bias_as_tuple(bias_value, 'index')
    bias_label = Label(f"C{chip_id}c{core_id}: " + BIAS_NAMES_SHORT[BIAS_TAGS_LOOKUP[bias_tag]],
                       layout=Layout(width='140px'))
    slider_handle = IntSlider(description='',
                              style=bias_slider_style,
                              min=0,
                              max=1811,
                              value=bias_value,
                              layout=Layout(width='300px'),
                              indent=False)
    tuple_label = Label(f"({bias_value_tuple[0]},{bias_value_tuple[1]})", layout=Layout(width='50px'),
                        indent=False)
    
    # Attach tuple printout updates to the value update of the slider
    
    def update_tuple_bias(new_value):
        """Parameter `tuple` format print updater for the parameter slider.

        Args:
            new_value (int): new slider value in `index` format.
        """
        coarse, fine = ut._bias_as_tuple(new_value['new'], 'index')
        tuple_label.value = f'({coarse},{fine})'
    
    slider_handle.observe(update_tuple_bias, names='value')
    
    widget_list = [bias_label, slider_handle, tuple_label]
    
    main_slider_box = HBox(widget_list, layout=Layout(display='flex', align_items="center"))
    
    clamping_box = None
    # clamp gain bias if the request bias is a leak
    bias_tag_attached = None
    
    # get the neuron gain if it is the neuron leak
    if bias_tag == "tau1":
        bias_tag_attached = 'gain'
    
    # get synapse type string if it is the synapse leak
    if bias_tag.rsplit("_", 1)[-1] == "tau":
        syn_type_str = bias_tag.rsplit("_", 1)[0]
        bias_tag_attached = syn_type_str + '_gain'
    
    # if it is one of the leaks, create additional bindings to the gain bias setter and create a checkbox
    if bias_tag_attached:
        clamp_gain_checkbox = Checkbox(
            value=False,
            description='Clamp gain',
            disabled=False,
            indent=False,
            layout=Layout(width='100px', align_items = 'flex-start')
        )
        
        fixed_ratio_box = FloatText(
            value=2,
            description='Ratio (linear): ',
            disabled=True,
            indent=False,
            layout=Layout(width='140px', align_items = 'flex-start')
        )
        
        def handle_checkbox_state_change(new_value):
            fixed_ratio_box.disabled = not new_value['new']
        
        clamp_gain_checkbox.observe(handle_checkbox_state_change, names='value')
        
        def update_gain_clamped_to_tau(new_tau_value):
            if clamp_gain_checkbox.value:
                tau_linear = ut._bias_as_linear(new_tau_value['new'], 'index')
                fixed_gain_ratio = fixed_ratio_box.value
                gain_linear = int(np.clip(fixed_gain_ratio*tau_linear, LINEAR_BIAS_MIN, LINEAR_BIAS_MAX))
                ut._set_bias(model, chip_id, core_id, BIAS_NAMES[BIAS_TAGS_LOOKUP[bias_tag_attached]], gain_linear, 'linear')
                
        slider_handle.observe(update_gain_clamped_to_tau, names='value')
        
        clamping_box = HBox([clamp_gain_checkbox, fixed_ratio_box], layout=Layout(display='flex', align_items = 'flex-start'))
            
    
    slider_bind = interactive_output(lambda x: ut._set_bias(model, chip_id, core_id, bias_name, x, 'index'), {'x':slider_handle})
    if clamping_box:
        return HBox([main_slider_box, clamping_box], layout=Layout(display='flex', align_items="flex-start"))
    else:
        return main_slider_box #, slider_bind
    
def make_parameter_slider_box(model, parameter_list : list, header_label='Parameter block'):
    """Creates a custom block of parameter slider from the user defined list.

    Args:
        model (Dynapse1Model): device model handle
        parameter_list (list): list of parameter tags to attach to sliders. Tags should contain a prefix 'Cxcx:' defining
                                target chip and core IDs.
        header_label (str, optional): Custom header string at the top of the block. Defaults to 'Parameter block'.

    Returns:
        _type_: _description_
    """
    slider_widgets = []
    slider_widgets_dict = {}
    #interactive_binders = []
    for parameter_tag in parameter_list:
        chip_id = int(parameter_tag.split(':')[0].split("C")[1].split('c')[0].strip())
        core_id = int(parameter_tag.split(':')[0].split("c")[1].strip())
        bias_tag = parameter_tag.split(':')[1]
        
        if bias_tag in BIAS_TAGS:
            slider_box = make_interactive_bias_slider(model, chip_id, core_id, bias_tag)
            slider_widgets.append(slider_box)
            slider_widgets_dict[parameter_tag] = slider_box
    
    link_clamped_gains_to_taus(model, slider_widgets_dict)
    
    header_label_widget = Label(header_label, style=label_style)
    return VBox([header_label_widget] + slider_widgets)

#def make_widget_

def make_function_button(function, text_label):
    button = Button(description=text_label, style=label_style)
    button.on_click(function)
    return button
    
def link_clamped_gains_to_taus(model, widget_dict):
    
    for parameter_tag in widget_dict.keys():
        if parameter_tag.split(':')[-1].split("_")[-1]=='gain':
            
            chip_id = int(parameter_tag.split(':')[0].split("C")[1].split('c')[0].strip())
            core_id = int(parameter_tag.split(':')[0].split("c")[1].strip())
            bias_tag = parameter_tag.split(':')[1]
            
            key_gain_box = parameter_tag
            
            # if it is neuron gain, use tau1 box, otherwise the corresponding synapse
            if bias_tag == 'gain':
                key_tau_box = key_gain_box.replace('gain', 'tau1')
            else:
                key_tau_box = key_gain_box.replace('gain', 'tau')
                
            if key_tau_box in widget_dict.keys():
                tau_slider = widget_dict[key_tau_box].children[0].children[1]
                tau_checkbox = widget_dict[key_tau_box].children[1].children[0]
                gain_slider = widget_dict[key_gain_box].children[1]
                
                link_disabled_state(gain_slider, tau_checkbox)
                link_gain_slider_state(model, parameter_tag, gain_slider, tau_slider, tau_checkbox)
                #def update_gain_value(new_gain_value):
                #    if clamp_gain_checkbox.value:
                #        tau_linear = ut._bias_as_linear(new_tau_value['new'], 'index')
                #        fixed_gain_ratio = fixed_ratio_box.value
                #        gain_linear = int(np.clip(fixed_gain_ratio*tau_linear, LINEAR_BIAS_MIN, LINEAR_BIAS_MAX))
                #        ut._get_bias(model, chip_id, core_id, BIAS_NAMES[BIAS_TAGS_LOOKUP[bias_tag_attached]], 'linear')
               # 

def link_disabled_state(gain_slider, checkbox):
    def change_gain_slider_state_by_checkbox(value):
        gain_slider.disabled = value['new']
                    
    checkbox.observe(change_gain_slider_state_by_checkbox, names = 'value')
    
def link_gain_slider_state(model, gain_parameter_tag, gain_slider, tau_slider, tau_checkbox):
    
    chip_id = int(gain_parameter_tag.split(':')[0].split("C")[1].split('c')[0].strip())
    core_id = int(gain_parameter_tag.split(':')[0].split("c")[1].strip())
    bias_tag = gain_parameter_tag.split(':')[1]
    
    def change_gain_slider_state_by_tau(value):
        if tau_checkbox.value:
            gain_bias_index = ut._get_bias(model, chip_id, core_id, BIAS_NAMES[BIAS_TAGS_LOOKUP[bias_tag]], 'index')
            gain_slider.value = gain_bias_index
                    
    tau_slider.observe(change_gain_slider_state_by_tau, names = 'value')

    
# # Gui Styling
# style = {'description_width': 'initial'}
# style_labels = {'font_weight' : 'bold'}


# # WTA_Gui setup
# sfa_label1 = Label('SFA demo setup', style=style_labels)

# sfa_label2 = Label('Main neuron parameters', style=style_labels)

# sfa_label3 = Label('Adaptation parameters', style=style_labels)


# # Create button to open the full bias slider window (if needed)
# def show_full_sliders_gui(b):
#     run_threaded_gui(model)
    
# button_show_bias_gui = Button(description='Full biases window', style=style_labels)
# button_show_bias_gui.on_click(show_full_sliders_gui)

# # Create button to send a regular rate pulse to demonstrate adaptation

# def single_pulse(b):
#     TG_SFA.set_regular_bump_input(0.5, sigma=10, amplitude=100)
#     sleep(0.5)
#     TG_SFA.regular_spike_gen.stop()
    
# button_single_pulse = Button(description='One 100 Hz Pulse (0.5s)', style=style_labels)
# button_single_pulse.on_click(single_pulse)

# def regular_input_on(b):
#     TG_SFA.set_regular_bump_input(0.5, sigma=10, amplitude=100)
    
# button_constant_input = Button(description='100 Hz constant input', style=style_labels)
# button_constant_input.on_click(regular_input_on)

# def ten_pulses(b):
#     for _ in range(10):
#         single_pulse(b)
#         sleep(1)
    
# button_ten_pulses = Button(description='10 pulses (100Hz, 0.5s)', style=style_labels)
# button_ten_pulses.on_click(ten_pulses)

# def record_single_response(b):
#     #from IPython.display import clear_output
#     #clear_output(wait=True)
#     #%matplotlib notebook
#     recording_reponse(TG_SFA.sink_node, TG_SFA.regular_spike_gen, TG_SFA.exc_size)
#     plt.show()
    
# button_adaptation_plot = Button(description='Plot pulse response', style=style_labels)
# button_adaptation_plot.on_click(record_single_response)

# # Input to Exc weight (NMDA)
# sfa_inp_bias = ut.get_nmda_weight(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider1 = IntSlider(description='Inp->E (NMDA)', style=style, min=0, max=1811, value=sfa_inp_bias)
# sfa_inp_wgt_slider = interactive_output(lambda x: ut.set_nmda_weight(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider1})
# # DC input bias
# sfa_neuron_dc_bias = ut.get_neuron_dc(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider2 = IntSlider(description='Core DC input', style=style, min=0, max=1811, value=sfa_neuron_dc_bias)
# sfa_ee_wgt_slider = interactive_output(lambda x: ut.set_neuron_dc(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider2})
# # Tau 1 bias
# sfa_tau1_bias = ut.get_neuron_tau1(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider3 = IntSlider(description='Tau 1', style=style, min=0, max=1811, value=sfa_tau1_bias)
# sfa_ie_wgt_slider = interactive_output(lambda x: ut.set_neuron_tau1(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider3})
# # Refractory period bias
# sfa_rfr_bias = ut.get_neuron_refractory_period(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider4 = IntSlider(description='Refractory period', style=style, min=0, max=1811, value=sfa_rfr_bias)
# sfa_ie_wgt_slider = interactive_output(lambda x: ut.set_neuron_refractory_period(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider4})
# # Neuron gain bias
# sfa_gain_bias = ut.get_neuron_gain(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider5 = IntSlider(description='Neuron gain', style=style, min=0, max=1811, value=sfa_gain_bias)
# sfa_ie_wgt_slider = interactive_output(lambda x: ut.set_neuron_gain(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider5})


# # Adaptation weight
# sfa_ahp_wgt_bias = ut.get_neuron_adaptation_weight(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider6 = IntSlider(description='SFA weight', style=style, min=0, max=1811, value=sfa_ahp_wgt_bias)
# sfa_ie_wgt_slider = interactive_output(lambda x: ut.set_neuron_adaptation_weight(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider6})
# # Adaptation tau
# sfa_ahp_tau_bias = ut.get_neuron_adaptation_tau(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider7 = IntSlider(description='SFA tau', style=style, min=0, max=1811, value=sfa_ahp_tau_bias)
# sfa_ie_tau_slider = interactive_output(lambda x: ut.set_neuron_adaptation_tau(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider7})
# # Adaptation weight
# sfa_ahp_gain_bias = ut.get_neuron_adaptation_gain(model, sfa_chip_id, sfa_core_id, 'index') # get the loaded value to the slider
# sfa_slider8 = IntSlider(description='SFA gain', style=style, min=0, max=1811, value=sfa_ahp_gain_bias)
# sfa_ie_gain_slider = interactive_output(lambda x: ut.set_neuron_adaptation_gain(model, sfa_chip_id, sfa_core_id, x, 'index'), {'x':sfa_slider8})


# SFA_label_box = VBox([sfa_label1, button_show_bias_gui, button_single_pulse, button_constant_input, button_ten_pulses])
# SFA_main_biases_box = VBox([sfa_label2, sfa_slider1, sfa_slider2, sfa_slider3, sfa_slider4, sfa_slider5])
# SFA_sfa_biases_box = VBox([sfa_label3, sfa_slider6, sfa_slider7, sfa_slider8, button_adaptation_plot])

# HBox([SFA_label_box, SFA_main_biases_box, SFA_sfa_biases_box], layout=Layout(grid_gap='10px 70px'))