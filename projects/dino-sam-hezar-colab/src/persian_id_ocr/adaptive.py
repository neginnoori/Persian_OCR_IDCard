"""Optional, auditable field repair. Never use expected identity values."""
from copy import deepcopy
import re
import unicodedata
from .anchors import assign_anchored_fields
from .anchor_gaps import supplement_name_rows
from .reader import normalize_text

NUMERIC = {'national_id', 'date_of_birth', 'expiry_date'}

def compatible(field, value):
    value = normalize_text(value or '')
    if field == 'national_id':
        return bool(re.fullmatch(r'[0-9]{10}', value))
    if field in NUMERIC:
        return bool(re.fullmatch(r'1[234][0-9]{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])', value))
    name_characters = all(c.isalpha() or unicodedata.category(c).startswith('M')
                          or c.isspace() or c in '\u200c\u200d' for c in value)
    return bool(value and any(c.isalpha() for c in value) and name_characters and
                not re.search(r'[0-9]', value) and
                not any(x in value.replace(' ', '') for x in ('شمارهثبت','کپی','شرکت','برابراصل')))


def repair_fields(image, regions, baseline, recognize=None):
    output = deepcopy(baseline)
    proposals, evidence = assign_anchored_fields(regions, image.size)
    proposals, gap_audit = supplement_name_rows(regions, proposals, evidence, compatible)
    audit = {'version': 'adaptive_fields_v2', 'baseline_fields': deepcopy(baseline),
             'anchor_evidence': evidence, 'row_recovery': gap_audit,
             'proposals': proposals, 'decisions': [], 'rereads': []}
    consistent = evidence.get('reason') == 'consistent_label_anchors'
    for row, (old, new, result) in enumerate(zip(baseline, proposals, output)):
        reason = 'baseline_preserved'
        same = old.get('value') == new.get('value')
        valid = new['status'] == 'candidate' and compatible(new['field'],new['value'])
        old_valid = old['status'] == 'candidate' and compatible(old['field'],old['value'])
        # An anchored value may repair a wrong template row, or remove an inline
        # label. A valid conflicting reading of the SAME region stays pending.
        old_ids = set(old.get('source_region_ids', []))
        new_ids = set(new.get('source_region_ids', []))
        wrong_region = bool(new_ids and not old_ids.intersection(new_ids))
        stripped_label = bool(new_ids == old_ids and new['value'] and
                              normalize_text(old.get('value') or '').endswith(new['value']))
        replace = consistent and valid and not same and (not old_valid or wrong_region or stripped_label)
        interpolated = new['assignment_method'] == 'interpolated_anchor_row_v1'
        if interpolated and old_valid:
            box = old.get('box')
            expected_y = gap_audit['row_origin'] + row * gap_audit['row_step']
            # Interpolation is weaker evidence than a readable title. It may
            # repair a displaced template row, but cannot overwrite a plausible
            # name already on this row, even if another region is proposed.
            displaced = box and abs((box[1] + box[3]) / 2 - expected_y) > .4 * gap_audit['row_step']
            replace = replace and bool(displaced)
        if replace:
            result.update(deepcopy(new))
            result['assignment_method'] = ('adaptive_interpolated_row_v1' if interpolated
                                           else 'adaptive_label_anchors_v1')
            reason = ('interpolated_row_replaces_template' if interpolated else
                      'anchored_region_replaces_template' if wrong_region else 'anchored_value_repair')
        elif consistent and valid and not same:
            reason = 'conflicting_candidate_preserved_for_review'
        audit['decisions'].append({'field':old['field'],'replaced':bool(replace),'reason':reason})
    # Local rereads are suggestions, not silent corrections of plausible names.
    # Invalid numeric values can be repaired only by two agreeing pure readings.
    if recognize:
        for item in output:
            value = item.get('value') or ''
            short_name = item['field'] not in NUMERIC and 0 < len(value.replace(' ','')) <= 2
            invalid = not compatible(item['field'], value)
            if not item.get('box') or not (short_name or invalid):
                continue
            x1,y1,x2,y2 = item['box']; h=y2-y1
            boxes = [[max(0,int(x1-.15*h)), max(0,int(y1-m*h)),
                      min(image.width,int(x2+.15*h)),min(image.height,int(y2+m*h))] for m in (.3,.5)]
            if any(b[2]<=b[0] or b[3]<=b[1] for b in boxes):
                continue
            texts = list(recognize([image.crop(tuple(b)) for b in boxes]))
            if len(texts)!=2:
                raise RuntimeError('Adaptive reread count mismatch')
            values = [normalize_text(t).strip() for t in texts]
            agreement = values[0] == values[1] and compatible(item['field'],values[0])
            accepted = agreement and invalid and item['field'] in NUMERIC
            audit['rereads'].append({'field':item['field'],'texts':texts,'values':values,
                                    'crop_boxes':boxes,'accepted':bool(accepted),
                                    'original_value':value,'original_box':item['box']})
            if accepted:
                item.update(value=values[0],status='candidate',review_required=True,
                            assignment_method=item['assignment_method']+'+local_consensus')
    # An ID/date left in a name slot is not a valid preserved name. Retain the
    # original field in the audit and raw text, while reporting the type error.
    for item, decision in zip(output, audit['decisions']):
        if item['field'] not in NUMERIC and item.get('value') and not compatible(item['field'], item['value']):
            item.update(value=None, status='invalid_format', review_required=True)
            decision.update(replaced=True, reason='invalid_name_value_rejected')
    return output, audit
