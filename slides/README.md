---
sort: 3
title: Slides
---

Here are all the slides from each topic in the class.


{%- for image in site.static_files -%}
{% if image.path contains '/slides/' %}
{% assign name = image.path | split: '/' %}
- [{{name[3]}}]({{site.baseurl}}{{image.path}})
{% endif %}
{%- endfor -%}
