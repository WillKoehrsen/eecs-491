c = get_config()

c.TemplateExporter.exclude_input_prompt = True
c.TemplateExporter.exclude_output_prompt = True
c.NbConvertApp.notebooks = ['assign_1.ipynb']
c.NbConvertApp.export_format = 'pdf'


