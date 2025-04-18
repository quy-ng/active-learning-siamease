from .character_level import *


def to_cuda(loader, device):
    """
    Transfer your dataloader into CPU or GPU
    @param loader: DataLoader
    @param device: torch.device
    @return: dataloader in specific device
    """
    return [load.to(device) for load in loader]


synonym_dict = {
    "name": [
        ["[^a-zA-Z]inc.", "[^a-zA-Z]inc"],
        ["[^a-zA-Z]co.", "[^a-zA-Z]co[^a-zA-Z]", "company"],
        ["ltd.", "ltd", "limited"],
        ["pvt.", "[^a-zA-Z]pvt", "private"],
        ["[^a-zA-Z]llc.", "[^a-zA-Z]llc"],
        ["[^a-zA-Z]no.", "[^a-zA-Z]no"],
        ["[^a-zA-Z]us[^a-zA-Z]", "[^a-zA-Z]usa[^a-zA-Z]", "united states"],
        ["[^0-9]3[^0-9]", "[^a-zA-Z0-9]iii[^a-zA-Z0-9]", "[^0-9]03[^0-9]"],
        ["[^0-9]2[^0-9]", "[^a-zA-Z0-9]ii[^a-zA-Z0-9]", "[^0-9]02[^0-9]"],
        ["[^0-9]1[^0-9]", "[^a-zA-Z0-9]i[^a-zA-Z0-9]", "[^0-9]01[^0-9]"],
    ],
    "address": [
        #         ["[^0-9]1[^0-9]", "[^0-9]01[^0-9]"],
        #         ["[^0-9]2[^0-9]", "[^0-9]02[^0-9]"],
        #         ["[^0-9]3[^0-9]", "[^0-9]03[^0-9]"],
        #         ["[^0-9]4[^0-9]", "[^0-9]04[^0-9]"],
        #         ["[^0-9]5[^0-9]", "[^0-9]05[^0-9]"],
        #         ["[^0-9]7[^0-9]", "[^0-9]06[^0-9]"],
        #         ["[^0-9]8[^0-9]", "[^0-9]08[^0-9]"],
        #         ["[^0-9]9[^0-9]", "[^0-9]09[^0-9]"],
        #         ["1st", "first"],
        #         ["2nd", "second"],
        #         ["3rd", "third"],
        #         ["4th", "fourth"],
        #         ["5th", "fifth"],
        [
            "[^a-zA-Z]k.[^a-zA-Z]",
            "[^a-zA-Z]kat.[^a-zA-Z]",
            "[^a-zA-Z]k:[^a-zA-Z]",
            "[^a-zA-Z]kat:[^a-zA-Z]",
            "[^a-zA-Z]k[^a-zA-Z]",
            "[^a-zA-Z]kat[^a-zA-Z]",
        ],
        ["[^a-zA-Z]area[^a-zA-Z]", "[^a-zA-Z]zone[^a-zA-Z]"],
        ["[^a-zA-Z]no-", "[^a-zA-Z]no.", "[^a-zA-Z]no:", "[^a-zA-Z]no[^a-zA-Z.-]"],
        ["country", "county"],
        ["road", "rd[.]{1}", "[^a-zA-Z0-9]rd[^a-zA-Z.]"],
        [
            "street",
            "[^a-zA-Z0-9]str[.]{1}",
            "[^a-zA-Z0-9]str[^a-zA-Z.]",
            "[^a-zA-Z0-9]st[.]{1}",
            "[^a-zA-Z0-9]st[^a-zA-Z.]",
        ],
        ["drive", "[^a-zA-Z]dr[^a-zA-Z.]", "[^a-zA-Z]dr[.]{1}"],
        ["avenue", "[^a-zA-Z]ave[.]{1}", "[^a-zA-Z]ave[^a-zA-Z.]"],
        ["boulevard", "[^a-zA-Z]blvd[.]{1}", "blvd[^.]"],
        ["lane", "[^a-zA-Z]ln[.]{1}", "[^a-zA-Z0-9]ln[^a-zA-Z.]"],
        ["sector[^-]", "sector-"],
        ["[^a-zA-Z]court[^a-zA-Z]", "[^a-zA-Z]ct[^a-zA-Z.]"],
        ["china", "[^a-zA-Z]cn[^a-zA-Z]", "c[.]{1}n"],
        ["united states", "u[.]{1}s", "[^a-zA-Z]us[^a-zA-Z]", "usa"],
        ["vietnam", "viet nam", "[^a-zA-Z]vn[^a-zA-Z]"],
    ],
}
