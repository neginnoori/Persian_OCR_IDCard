"""Conservative numeric-only rereading for merged label/value regions."""
import re
from statistics import median
from .reader import normalize_text


def retry_numeric_fields(image, fields, recognize, *, frame=None):
    left, top, right, bottom = frame or (0, 0, *image.size)
    width = right - left
    # Infer the value-column boundary from other detected values, not answers.
    anchors = [f['box'][2] for f in fields if f['box'] and
               f['field'] in ('first_name', 'last_name', 'father_name') and
               .65 <= (f['box'][2]-left)/width <= .83]
    if len(anchors) < 2:
        return fields
    boundary = median(anchors) + .008 * width
    output = []
    for original in fields:
        item = dict(original)
        box = item['box']
        if (item['field'] not in ('national_id', 'date_of_birth', 'expiry_date')
                or not box or (box[2]-left)/width <= .85):
            output.append(item)
            continue
        x1, y1, _, y2 = box
        h = y2-y1
        if boundary <= x1 or h <= 0:
            output.append(item)
            continue
        crop_boxes = [[max(0, int(x1-.2*h)), max(0, int(y1-margin*h)),
                       min(image.width, int(boundary)), min(image.height, int(y2+margin*h))]
                      for margin in (.3, .6)]
        raws = recognize([image.crop(tuple(b)) for b in crop_boxes])
        if len(raws) != 2:
            raise RuntimeError('Numeric retry recognition count mismatch')
        values = [re.sub(r'\s+', '', normalize_text(t)) for t in raws]
        pattern = (r'[0-9]{10}' if item['field'] == 'national_id' else
                   r'1[234][0-9]{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])')
        valid_indices = [i for i, value in enumerate(values) if re.fullmatch(pattern, value)]
        valid_values = {values[i] for i in valid_indices}
        accepted = len(valid_values) == 1
        selected = valid_indices[0] if accepted else None
        reason = ('agreement' if accepted and len(valid_indices) == 2 else
                  'single_valid_candidate' if accepted else 'no_unique_valid_candidate')
        if accepted and item['status'] == 'candidate' and values[selected] != item['value']:
            accepted = False
            reason = 'conflicts_with_original_candidate'
        item['numeric_retry'] = {'method': 'value_column_crop_two_margins_v1',
                                 'texts': list(raws), 'values': values,
                                 'crop_boxes': crop_boxes, 'accepted': accepted,
                                 'selection_reason': reason,
                                 'selected_index': selected if accepted else None,
                                 'original_value': item['value'], 'original_box': box}
        if accepted:
            item.update(value=values[selected], status='candidate', box=crop_boxes[selected],
                        assignment_method=item['assignment_method']+'+numeric_retry')
        # Raw detector text and source IDs remain untouched; agreement is not proof.
        item['review_required'] = True
        output.append(item)
    return output
