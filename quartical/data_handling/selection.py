def filter_xds_list(xds_list, fields, ddids):

    filter_fields = {"FIELD_ID": fields,
                     "DATA_DESC_ID": ddids}

    for k, v in filter_fields.items():
        fil = filter(lambda xds: getattr(xds, k) in v, xds_list)
        xds_list = list(fil) if v else xds_list

    if len(xds_list) == 0:
        raise ValueError("Selection of field/ddid has deselected all data.")

    return xds_list
