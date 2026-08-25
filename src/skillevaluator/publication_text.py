# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Unicode security normalization for publication identities."""

from __future__ import annotations

import math
import unicodedata

UNICODE_CONFUSABLES_VERSION = "17.0.0"
UNICODE_CONFUSABLES_SOURCE_SHA256 = "091c7f82fc39ef208faf8f94d29c244de99254675e09de163160c810d13ef22a"
UNICODE_CONFUSABLES_GENERATOR_UCD_VERSION = "15.1.0"
UNICODE_CONFUSABLES_SUBSET_SOURCE_COUNT = 1954

_INVISIBLE_IDENTITY_CHARACTERS = frozenset(
    {
        "\u115f",
        "\u1160",
        "\u2800",
        "\u3164",
        "\uffa0",
        "\U00013441",
        "\U00013442",
        "\U0001d159",
    }
)
_RESERVED_IDENTITIES = frozenset(
    {
        "model not recorded",
        "model unknown",
        "n a",
        "na",
        "none",
        "not applicable",
        "not available",
        "not recorded",
        "null",
        "tbd",
        "unknown",
        "unknown model",
        "unset",
    }
)
_SECURITY_TEXT_CONFUSABLES = {"\u0406": "I", "\u0456": "i"}

# Generated from the Unicode 17.0.0 confusables data. Each source maps directly
# to its case-insensitive effective prototype, retaining separator boundaries
# and sources whose prototype disappears under the pinned filtering contract.
# This is the complete subset relevant to the reserved identities above;
# keeping that closed subset avoids carrying the full Unicode security table.
# Generation pins Python's Unicode Character Database 15.1.0 so future host
# category changes do not silently alter the committed map.
# Source: https://www.unicode.org/Public/17.0.0/security/confusables.txt
# Unicode Data Files and Software License: https://www.unicode.org/license.txt
_CONFUSABLE_PROTOTYPE_GROUPS = (
    (
        "",
        "\u05ad\u05ae\u05a8\u05a4\u1ab4\u20db\u0619\u08f3\u0343\u0315\u064f\u065d\u059c\u059d\u0618\u0747\u0341"
        "\u0954\u064e\u0340\u0953\u030c\ua67c\u0658\u065a\u036e\u0945\U00011b66\u06e8\u0310\u0901\u0981\u0a81\u0b01"
        "\u0c00\u0c81\u0d01\U000114bf\u1cd0\u0311\u065b\u07ee\ua6f0\u05af\u06df\u17d3\u309a\u0652\u0b82\u1036\u17c6"
        "\U00011300\u0e4d\u0ecd\u0366\u2dea\u08eb\u07f3\u064b\u08f0\u0342\u0653\u05c4\u06ec\u0740\u08ea\u0741\u0358"
        "\u05b9\u05ba\u05c2\u05c1\u07ed\u0902\u0a02\u0a82\u0bcd\u0337\u1ab7\u0322\u0345\u1cd2\u0305\u0659\u07eb"
        "\ua6f1\u1ae2\u1ae8\u1cda\u0657\u0357\u08ff\u08f8\u0900\u1ad9\U0001e6ee\u1ced\u1cdc\u0656\u1cd5\u0347\u08f9"
        "\u08fa\u309b\u309c\u0336\u302c\u05c5\u08ed\u1cdd\u05b4\u065c\u093c\u09bc\u0a3c\u0abc\u0b3c\U000111ca"
        "\U000114c3\U00010a3a\u08ee\u1cde\u0f37\u302d\u0327\u0321\u0339\u1cd9\u1cd8\u0952\u0320\u08f1\u08e8\u08e5"
        "\u08f2\u061a\u0317\u065f\u030d\u0742\u0a03\u0c03\u0c83\u0d03\u0d83\u1038\U000114c1\u17cb\u0ec8\u0ec9\u0eca"
        "\u0ecb\ua66f\u2df6\u2ded\u2df7\u2de8\u2def\u1dee\u0949\u093b\U000111cb\U00011b60\u0ac1\u0ac2\u0a4b\u0a48"
        "\u0a4d\u0acd\U000114b0\u093f\u0a3f\U000114b1\U000114b9\U000114bc\U000114be\U000114c2\U000114bd\u1031\u0d3f"
        "\u0d40\u0d46\u0d48\u0d47\u0cbf\u0cc1\u0c42\u0cc3\u0c44\u0d42\u0d43\U000115dc\U000115dd\u17b7\u17b8\u17b9"
        "\u17ba\u0eb8\u0eb9\u0f77\u0f79\u0f7b\u0f7d\U00011cb2\u1734\u109e\u3164\u0cdc\u1de8\u2dee\u1ae7\u031a\u0295"
        "\ua7cf\u0348\u0956\u0a41\u0957\u0a42\u0947\u0a47\u5152\U0001f40d\U0001f443\U0001f377\U0001f3e2\U0001f333"
        "\U0001f34e\U0001f34f\U0001f352\U0001f353\u28ff\u29b5\u21c4\u21cc\u2657\u265d\U0001f514\u6138",
    ),
    (
        " ",
        "\ufc5e\ufc5f\ufc60\ufc61\ufc62\ufc63\u2028\u2029\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2008"
        "\u2009\u200a\u205f\xa0\u2007\u202f\u07fa\ufe4d\ufe4e\ufe4f\u2010\u2011\u2012\u2013\ufe58\u06d4\u2043\u02d7"
        "\u2212\u2796\u2cbb\u2cba\u2a29\u2e1a\ufb29\u2238\u2cb3\u2cb2\u2a2a\ua4fe\uff5e\u060d\u066b\u201a\xb8\ua4f9"
        "\u2e32\u066c\u037e\u2e35\u0903\u0a83\uff1a\u0589\u0703\u0704\u16ec\ufe30\u1803\u1809\u205a\u05c3\u02f8"
        "\ua789\u2236\u02d0\ua4fd\U00011dd9\u2a74\u29f4\uff01\u01c3\u2d51\u203c\u2049\u0294\u0241\u097d\u13ae\ua6eb"
        "\u2048\u2047\u2e2e\U0001d16d\u2024\u0701\u0702\ua60e\U00010a50\u0660\u06f0\ua4f8\ua4fb\u2025\ua4fa\u2026"
        "\ua6f4\u30fb\uff65\u16eb\u0387\u2e31\U00010101\u2022\u2027\u2219\u22c5\ua78f\u1427\u22ef\u2d48\u1444\u22d7"
        "\u1437\u1440\ua830\u0965\u1c3c\u104b\u1aa9\u1aab\u1b5f\U00010a57\U0001144c\U00011642\U00011c42\u1c7f\u055d"
        "\uff07\u2018\u2019\u201b\u05f3\u2032\u2035\u055a`\u1fef\uff40\xb4\u0384\u1ffd\u1fbd\u1fbf\u1ffe\u02b9"
        "\u0374\u02c8\u02ca\u02cb\u02f4\u02bb\u02bd\u02bc\u02be\ua78c\u05d9\u07f4\u07f5\u144a\u16cc\U00016f51"
        '\U00016f52\u1cd3"\uff02\u201c\u201d\u201f\u05f4\u2033\u2036\u3003\u02dd\u02ba\u02f6\u02ee\u05f2\u2034'
        "\u2037\u2057\uff3b\u2768\u2772\u3014\ufd3e\u2e28\uff3d\u2769\u2773\u3015\ufd3f\u2e29\u2774\U0001d114\u2775"
        "\u301a\u301b\u27e8\u2329\u3008\u31db\u304f\U00021fe8\u27e9\u232a\u3009\uff3e\u2e3f\u204e\u066d\u2217"
        "\U0001031f\u1735\u2041\u2215\u2044\u2571\u27cb\u29f8\U0001d23a\u31d3\u3033\u2cc7\u2cc6\u30ce\u4e3f\u2f03"
        "\u29f6\u2afd\u2afb\uff3c\ufe68\u2216\u27cd\u29f5\u29f9\U0001d20f\U0001d23b\u31d4\u4e36\u2f02\u2cf9\u244a"
        "\ua778\u0af0\U000110bb\U000111c7\u26ac\U000111db\u17d9\u17d5\u17da\u0f0c\u0f0e\u02c4\u02c6\u02fb\ua716"
        "\ua714\u3002\u2e30\u02da\u2218\u25cb\u25e6\u235c\U00010ed0\u2364\u0bf5\u0f1b\u0f1f\u0fce\u0f1e\u24b8\u24c7"
        "\u24c5\U0001d21b\u2bec\u2bed\u2bee\u2bef\u21b5\u2965\U0001d6db\U0001d715\U0001d74f\U0001d789\U0001d7c3"
        "\U0001e8cc\U0001e8cd\xf0\u2300\U0001d6c1\U0001d6fb\U0001d735\U0001d76f\U0001d7a9\U000118a8\u2362\u236b"
        "\u2588\u25a0\u2a3f\u16ed\u2795\U0001029b\U0001e6e9\u2a23\u2a22\u2a24\u2214\u2a25\u2a26\u2797\u2039\u276e"
        "\u02c2\U0001d236\u1438\u16b2\u22d6\u2cb5\u2cb4\u1445\u226a\u22d8\u1400\u2e40\u30a0\ua4ff\u225a\u2259\u2257"
        "\u2250\u2251\u2b96\u2a6e\u2a75\u2a76\u225e\u203a\u276f\u02c3\U0001d237\u1433\U00016f3f\u1441\u2aa5\u226b"
        "\u2a20\u22d9\u2053\u02dc\u1fc0\u223c\u2368\u2e1e\u2a6a\u2e1f\U0001e8c8\u22c0\u222f\u2230\u2e2b\u2e2a\u2e2c"
        "\U000111de\u264e\U0001f75e\u2263\u2cb7\u2a03\u2a04\U0001d238\U0001d239\u2a05\u2a06\u2a02\u235f\U0001f771"
        "\U0001f755\u25c1\u25b7\u2363\ufe34\u25e0\u2a3d\u2325\u29c7\u25ce\u29be\u29c5\u29b0\u23c3\u23c2\u23c1\u23c6"
        "\u2638\ufe35\ufe36\ufe37\ufe38\ufe39\ufe3a\u25b1\u23fc\ufe31\uff5c\u2503\u250f\u2523\u2590\u2597\u259d"
        "\u2610\uffed\u25b8\u25ba\u2ce9\U0001f70a\U0001f312\U0001f319\u23fe\U0001f318\u29d9\U0001f73a\u2a3e\u2669"
        "\u266a\u24ea\u21ba\U0001ccfb\U0001f10f\u20a4\u3012\u3036\ufb39",
    ),
    (
        "a",
        "\u237a\uff41\U0001d41a\U0001d44e\U0001d482\U0001d4b6\U0001d4ea\U0001d51e\U0001d552\U0001d586\U0001d5ba"
        "\U0001d5ee\U0001d622\U0001d656\U0001d68a\u0251\u03b1\U0001d6c2\U0001d6fc\U0001d736\U0001d770\U0001d7aa"
        "\u0430\uff21\U0001ccd6\U0001d400\U0001d434\U0001d468\U0001d49c\U0001d4d0\U0001d504\U0001d538\U0001d56c"
        "\U0001d5a0\U0001d5d4\U0001d608\U0001d63c\U0001d670\u0391\U0001d6a8\U0001d6e2\U0001d71c\U0001d756\U0001d790"
        "\u0410\u13aa\u15c5\ua4ee\U00016f40\U000102a0\u2376\u01ce\u01cd\u0227\u0226\u1e9a",
    ),
    (
        "b",
        "\U0001d41b\U0001d44f\U0001d483\U0001d4b7\U0001d4eb\U0001d51f\U0001d553\U0001d587\U0001d5bb\U0001d5ef"
        "\U0001d623\U0001d657\U0001d68b\u0184\u042c\u13cf\u1472\u15af\U00016eb6\uff22\u212c\U0001ccd7\U0001d401"
        "\U0001d435\U0001d469\U0001d4d1\U0001d505\U0001d539\U0001d56d\U0001d5a1\U0001d5d5\U0001d609\U0001d63d"
        "\U0001d671\ua7b4\u0392\U0001d6a9\U0001d6e3\U0001d71d\U0001d757\U0001d791\u2c82\u0412\u13f4\u15f7\ua4d0"
        "\U00010282\U000102a1\U00010301\u0253\u1473\u0183\u0182\u0411\u0180\u048d\u048c\u0463\u0462",
    ),
    (
        "c",
        "\uff43\u217d\U0001d41c\U0001d450\U0001d484\U0001d4b8\U0001d4ec\U0001d520\U0001d554\U0001d588\U0001d5bc"
        "\U0001d5f0\U0001d624\U0001d658\U0001d68c\u1d04\u03f2\u2ca5\u0441\u1004\u105a\uabaf\U0001043d\U0001f74c"
        "\U000118e9\U000118f2\uff23\u216d\u2102\u212d\U0001ccd8\U0001d402\U0001d436\U0001d46a\U0001d49e\U0001d4d2"
        "\U0001d56e\U0001d5a2\U0001d5d6\U0001d60a\U0001d63e\U0001d672\u03f9\u2ca4\u0421\u13df\ua4da\U000102a2"
        "\U00010302\U00010415\U0001051c\xa2\u023c\u20a1\U0001f16e\xe7\u04ab\xc7\u04aa",
    ),
    (
        "d",
        "\u217e\u2146\U0001d41d\U0001d451\U0001d485\U0001d4b9\U0001d4ed\U0001d521\U0001d555\U0001d589\U0001d5bd"
        "\U0001d5f1\U0001d625\U0001d659\U0001d68d\u0501\u13e7\u146f\ua4d2\u216e\u2145\U0001ccd9\U0001d403\U0001d437"
        "\U0001d46b\U0001d49f\U0001d4d3\U0001d507\U0001d53b\U0001d56f\U0001d5a3\U0001d5d7\U0001d60b\U0001d63f"
        "\U0001d673\u13a0\u15de\u15ea\ua4d3\u0257\u0256\u018c\u0111\u0110\xd0\u0189\u20ab",
    ),
    (
        "e",
        "\u212e\uff45\u212f\u2147\U0001d41e\U0001d452\U0001d486\U0001d4ee\U0001d522\U0001d556\U0001d58a\U0001d5be"
        "\U0001d5f2\U0001d626\U0001d65a\U0001d68e\uab32\u0435\u04bd\u22ff\uff25\u2130\U0001ccda\U0001d404\U0001d438"
        "\U0001d46c\U0001d4d4\U0001d508\U0001d53c\U0001d570\U0001d5a4\U0001d5d8\U0001d60c\U0001d640\U0001d674\u0395"
        "\U0001d6ac\U0001d6e6\U0001d720\U0001d75a\U0001d794\u0415\u2d39\u13ac\ua4f0\U000118a6\U000118ae\U00010286"
        "\u011b\u011a\u0247\u0246\u04bf\u04be",
    ),
    (
        "i",
        "\u02db\u2373\uff49\u2170\u2139\u2148\U0001d422\U0001d456\U0001d48a\U0001d4be\U0001d4f2\U0001d526\U0001d55a"
        "\U0001d58e\U0001d5c2\U0001d5f6\U0001d62a\U0001d65e\U0001d692\u0131\U0001d6a4\u026a\u0269\u03b9\u1fbe\u037a"
        "\U0001d6ca\U0001d704\U0001d73e\U0001d778\U0001d7b2\u2c93\u0456\ua647\u0582\uab75\u13a5\U000118c3\u24db"
        "\u2378\u01d0\u0268\u1d7b\u1d7c",
    ),
    (
        "k",
        "\U0001d424\U0001d458\U0001d48c\U0001d4c0\U0001d4f4\U0001d528\U0001d55c\U0001d590\U0001d5c4\U0001d5f8"
        "\U0001d62c\U0001d660\U0001d694\u212a\uff2b\U0001cce0\U0001d40a\U0001d43e\U0001d472\U0001d4a6\U0001d4da"
        "\U0001d50e\U0001d542\U0001d576\U0001d5aa\U0001d5de\U0001d612\U0001d646\U0001d67a\u039a\U0001d6b1\U0001d6eb"
        "\U0001d725\U0001d75f\U0001d799\u2c94\u041a\u13e6\u16d5\ua4d7\U00010518\u0199\u2c69\u049a\u20ad\ua740\u049e",
    ),
    (
        "l",
        "\u01cf\u05c0|\u2223\u23fd\uffe81\u0661\u06f1\U00010320\U0001e8c7\U0001ccf1\U0001d7cf\U0001d7d9\U0001d7e3"
        "\U0001d7ed\U0001d7f7\U0001fbf1I\uff29\u2160\u2110\u2111\U0001ccde\U0001d408\U0001d43c\U0001d470\U0001d4d8"
        "\U0001d540\U0001d574\U0001d5a8\U0001d5dc\U0001d610\U0001d644\U0001d678\u0196\uff4c\u217c\u2113\U0001d425"
        "\U0001d459\U0001d48d\U0001d4c1\U0001d4f5\U0001d529\U0001d55d\U0001d591\U0001d5c5\U0001d5f9\U0001d62d"
        "\U0001d661\U0001d695\u01c0\u0399\U0001d6b0\U0001d6ea\U0001d724\U0001d75e\U0001d798\u2c92\u0406\u04cf\u04c0"
        "\u05d5\u05df\u0627\U0001ee00\U0001ee80\ufe8e\ufe8d\u07ca\u2d4f\u16c1\ua4f2\U00016f28\U0001028a\U00010309"
        "\U00011dda\U00011de1\U00016eaa\U0001d22a\u216c\u2112\U0001cce1\U0001d40b\U0001d43f\U0001d473\U0001d4db"
        "\U0001d50f\U0001d543\U0001d577\U0001d5ab\U0001d5df\U0001d613\U0001d647\U0001d67b\u2cd0\u13de\u14aa\ua4e1"
        "\U00016f16\U000118a3\U000118b2\U0001041b\U00010526\ufd3c\ufd3d\u0142\u0141\u026d\u0197\u019a\u026b\u0625"
        "\ufe88\ufe87\u0673\u0623\ufe82\ufe81",
    ),
    (
        "n",
        "\U0001d427\U0001d45b\U0001d48f\U0001d4c3\U0001d4f7\U0001d52b\U0001d55f\U0001d593\U0001d5c7\U0001d5fb"
        "\U0001d62f\U0001d663\U0001d697\u0578\u057c\uff2e\u2115\U0001cce3\U0001d40d\U0001d441\U0001d475\U0001d4a9"
        "\U0001d4dd\U0001d511\U0001d579\U0001d5ad\U0001d5e1\U0001d615\U0001d649\U0001d67d\u039d\U0001d6b4\U0001d6ee"
        "\U0001d728\U0001d762\U0001d79c\u2c9a\ua4e0\U00010513\U0001018e\u0273\u019e\u014b\u03b7\U0001d6c8\U0001d702"
        "\U0001d73c\U0001d776\U0001d7b0\u0572\u019d\u1d70\u0146\u2229\u22c2\U0001d245\u1260\u144e\ua4f5",
    ),
    (
        "o",
        "\u0c02\u0c82\u0d02\u0d82\u0966\u09e6\u0a66\u0ae6\u0b66\u0be6\u0c66\u0d66\u0e50\u0ed0\u1040\u17e0\U000114d0"
        "\u0665\u06f5\uff4f\u2134\U0001d428\U0001d45c\U0001d490\U0001d4f8\U0001d52c\U0001d560\U0001d594\U0001d5c8"
        "\U0001d5fc\U0001d630\U0001d664\U0001d698\u1d0f\u1d11\uab3d\u03bf\U0001d6d0\U0001d70a\U0001d744\U0001d77e"
        "\U0001d7b8\u03c3\U0001d6d4\U0001d70e\U0001d748\U0001d782\U0001d7bc\u2c9f\u03ed\u043e\u10ff\u0585\u05e1"
        "\u0647\U0001ee24\U0001ee64\U0001ee84\ufeeb\ufeec\ufeea\ufee9\u06be\ufbac\ufbad\ufbab\ufbaa\u06c1\ufba8"
        "\ufba9\ufba7\ufba6\u06d5\u0d20\u101d\U000104ea\U000118c8\U000118d7\U0001042c0\u07c0\u0ce6\u3007\U000118e0"
        "\U0001ccf0\U0001d7ce\U0001d7d8\U0001d7e2\U0001d7ec\U0001d7f6\U0001fbf0\uff2f\U0001cce4\U0001d40e\U0001d442"
        "\U0001d476\U0001d4aa\U0001d4de\U0001d512\U0001d546\U0001d57a\U0001d5ae\U0001d5e2\U0001d616\U0001d64a"
        "\U0001d67e\u039f\U0001d6b6\U0001d6f0\U0001d72a\U0001d764\U0001d79e\u2c9e\u041e\u0555\u2d54\u12d0\u0b20"
        "\U000104c2\ua4f3\U000118b5\U00010292\U000102ab\U00010404\U00010516\U00011de0\u2070\u1d52\u01d2\u01d1\u06ff"
        "\u0150\xf8\uab3e\xd8\u2d41\u01fe\u0275\ua74b\u2c91\u04e9\u0473\uab8e\uabbb\u2296\u229d\u236c\U0001d21a"
        "\U0001f714\u019f\ua74a\u03b8\u03d1\U0001d6c9\U0001d6dd\U0001d703\U0001d717\U0001d73d\U0001d751\U0001d777"
        "\U0001d78b\U0001d7b1\U0001d7c5\u0398\u03f4\U0001d6af\U0001d6b9\U0001d6e9\U0001d6f3\U0001d723\U0001d72d"
        "\U0001d75d\U0001d767\U0001d797\U0001d7a1\u2c90\u04e8\u0472\u2d31\u13be\u13eb\uab74\ufcd9\u01a1\u01a0\u10d7"
        "\u1010\u03db\U0001d6d3\U0001d70d\U0001d747\U0001d781\U0001d7bb\u2c8b\xf6\u06c2\ufba5\ufba4",
    ),
    (
        "p",
        "\u2374\uff50\U0001d429\U0001d45d\U0001d491\U0001d4c5\U0001d4f9\U0001d52d\U0001d561\U0001d595\U0001d5c9"
        "\U0001d5fd\U0001d631\U0001d665\U0001d699\xfe\u01bf\u03c1\u03f1\U0001d6d2\U0001d6e0\U0001d70c\U0001d71a"
        "\U0001d746\U0001d754\U0001d780\U0001d78e\U0001d7ba\U0001d7c8\u03f8\u2ca3\u2ccf\u0440\uff30\u2119\U0001cce5"
        "\U0001d40f\U0001d443\U0001d477\U0001d4ab\U0001d4df\U0001d513\U0001d57b\U0001d5af\U0001d5e3\U0001d617"
        "\U0001d64b\U0001d67f\u03a1\U0001d6b8\U0001d6f2\U0001d72c\U0001d766\U0001d7a0\u2ca2\u2cce\u0420\u13e2\u146d"
        "\ua4d1\U00010295\u01a5\u1d7d\u03f7\U000104c4",
    ),
    (
        "r",
        "\U0001d42b\U0001d45f\U0001d493\U0001d4c7\U0001d4fb\U0001d52f\U0001d563\U0001d597\U0001d5cb\U0001d5ff"
        "\U0001d633\U0001d667\U0001d69b\uab47\uab48\u1d26\u2c85\u0433\uab81\U0001d216\u211b\u211c\u211d\U0001cce7"
        "\U0001d411\U0001d445\U0001d479\U0001d4e1\U0001d57d\U0001d5b1\U0001d5e5\U0001d619\U0001d64d\U0001d681\u01a6"
        "\u13a1\u13d2\U000104b4\u1587\ua4e3\U00016f35\u027d\u027c\u024d\u0493\u1d72",
    ),
    (
        "s",
        "\uff53\U0001d42c\U0001d460\U0001d494\U0001d4c8\U0001d4fc\U0001d530\U0001d564\U0001d598\U0001d5cc\U0001d600"
        "\U0001d634\U0001d668\U0001d69c\ua731\u01bd\u0455\u0d1f\uabaa\U000118c1\U00010448\uff33\U0001cce8\U0001d412"
        "\U0001d446\U0001d47a\U0001d4ae\U0001d4e2\U0001d516\U0001d54a\U0001d57e\U0001d5b2\U0001d5e6\U0001d61a"
        "\U0001d64e\U0001d682\u0405\u054f\u13d5\u13da\ua4e2\U00016f3a\U00010296\U00010420\u0282\u1d74",
    ),
    (
        "t",
        "\U0001d42d\U0001d461\U0001d495\U0001d4c9\U0001d4fd\U0001d531\U0001d565\U0001d599\U0001d5cd\U0001d601"
        "\U0001d635\U0001d669\U0001d69d\u22a4\u27d9\U0001f768\uff34\U0001cce9\U0001d413\U0001d447\U0001d47b"
        "\U0001d4af\U0001d4e3\U0001d517\U0001d54b\U0001d57f\U0001d5b3\U0001d5e7\U0001d61b\U0001d64f\U0001d683\u03a4"
        "\U0001d6bb\U0001d6f5\U0001d72f\U0001d769\U0001d7a3\u2ca6\u0422\u13a2\ua4d4\U00016f0a\U000118bc\U00010297"
        "\U000102b1\U00010315\u01ad\u2361\u023e\u021a\u01ae\u04ac\u20ae\u0167\u0166\u1d75\u0163\u021b",
    ),
    (
        "u",
        "\U0001d42e\U0001d462\U0001d496\U0001d4ca\U0001d4fe\U0001d532\U0001d566\U0001d59a\U0001d5ce\U0001d602"
        "\U0001d636\U0001d66a\U0001d69e\ua79f\u1d1c\uab4e\uab52\u028b\u03c5\U0001d6d6\U0001d710\U0001d74a\U0001d784"
        "\U0001d7be\u057d\U000104f6\U000118d8\u222a\u22c3\U0001ccea\U0001d414\U0001d448\U0001d47c\U0001d4b0"
        "\U0001d4e4\U0001d518\U0001d54c\U0001d580\U0001d5b4\U0001d5e8\U0001d61c\U0001d650\U0001d684\u054d\u1200"
        "\U000104ce\u144c\ua4f4\U00016f42\U000118b8\u01d4\u01d3\u045f\u1d7e\uab9c\u0244\u13cc",
    ),
    (
        "v",
        "\u2228\u22c1\uff56\u2174\U0001d42f\U0001d463\U0001d497\U0001d4cb\U0001d4ff\U0001d533\U0001d567\U0001d59b"
        "\U0001d5cf\U0001d603\U0001d637\U0001d66b\U0001d69f\u1d20\u03bd\U0001d6ce\U0001d708\U0001d742\U0001d77c"
        "\U0001d7b6\u0475\u05d8\U00011706\uaba9\U000118c0\U0001d20d\u0667\u06f7\u2164\U0001cceb\U0001d415\U0001d449"
        "\U0001d47d\U0001d4b1\U0001d4e5\U0001d519\U0001d54d\U0001d581\U0001d5b5\U0001d5e9\U0001d61d\U0001d651"
        "\U0001d685\u0474\u2d38\u13d9\u142f\ua6df\ua4e6\U00016f08\U000118a0\U0001051d\U00010197\U0001f708",
    ),
    (
        "w",
        "\u026f\U0001d430\U0001d464\U0001d498\U0001d4cc\U0001d500\U0001d534\U0001d568\U0001d59c\U0001d5d0\U0001d604"
        "\U0001d638\U0001d66c\U0001d6a0\u1d21\u2cbd\u0461\u0448\u051d\u0561\U0001170a\U0001170e\U0001170f\uab83"
        "\U000118e6\U000118ef\U0001ccec\U0001d416\U0001d44a\U0001d47e\U0001d4b2\U0001d4e6\U0001d51a\U0001d54e"
        "\U0001d582\U0001d5b6\U0001d5ea\U0001d61e\U0001d652\U0001d686\u051c\u13b3\u13d4\ua4ea\u047d\U000114c5\u20a9"
        "\ua761\U0001d222\u13c7\u15ef\u047c\u2cbc",
    ),
    (" b", "\u147e\u1480\u0181"),
    (" c", "\u2103"),
    (" d", "\u147a\u018a"),
    (" l", "\u14b6"),
    (" n", "\u1459\u0149"),
    (" p", "\u1476\u01a4"),
    (" t", "\u01ac"),
    (" u", "\u1457"),
    (" v", "\u143a"),
    ("av", "\ua739\ua73b\ua738\ua73a"),
    ("b ", "\u147f\u1481\u1488"),
    ("bl", "\u042b"),
    ("c ", "\u0187"),
    ("d ", "\u147b\u1487"),
    ("k ", "\u0198"),
    ("l ", "\u0140\u013f\u14b7\U0001f102\u2488\u05f1"),
    ("ll", "\u2016\u2225\u2161\u01c1\u05f0\U00010199"),
    ("n ", "\u145a\u1468"),
    ("no", "\u2116"),
    ("o ", "\U0001f101\U0001f100\u13a4"),
    ("p ", "\u1477\u1486"),
    ("r ", "\u0491"),
    (
        "rn",
        "\uff2d\u216f\u2133\U0001cce2\U0001d40c\U0001d440\U0001d474\U0001d4dc\U0001d510\U0001d544\U0001d578"
        "\U0001d5ac\U0001d5e0\U0001d614\U0001d648\U0001d67c\u039c\U0001d6b3\U0001d6ed\U0001d727\U0001d761\U0001d79b"
        "\u03fa\u2c98\u041c\u13b7\u15f0\u16d6\ua4df\U000102b0\U00010311\u04cd\U000118e3m\u217f\U0001d426\U0001d45a"
        "\U0001d48e\U0001d4c2\U0001d4f6\U0001d52a\U0001d55e\U0001d592\U0001d5c6\U0001d5fa\U0001d62e\U0001d662"
        "\U0001d696\U00011700\u20a5\u0271\u1d6f\u1e43",
    ),
    ("u ", "\u1458\u1467"),
    ("v ", "\u143b"),
    ("w ", "\u18ed"),
    (" a ", "\u249c\U0001f110"),
    (" b ", "\u249d\U0001f111"),
    (" c ", "\u249e\U0001f112"),
    (" d ", "\u249f\U0001f113"),
    (" e ", "\u24a0\U0001f114"),
    (" i ", "\u24a4"),
    (" k ", "\u24a6\U0001f11a"),
    (" l ", "\u2474\U0001f118\u24a7\U0001f11b"),
    (" n ", "\u24a9\U0001f11d"),
    (" o ", "\u24aa\U0001f11e"),
    (" p ", "\u24ab\U0001f11f"),
    (" r ", "\u24ad\U0001f121"),
    (" s ", "\u24ae\U0001f122\U0001f12a"),
    (" t ", "\u24af\U0001f123"),
    (" u ", "\u24b0\U0001f124"),
    (" v ", "\u24b1\U0001f125"),
    (" w ", "\u24b2\U0001f126"),
    ("ll ", "\u2492"),
    (" ll ", "\u247e"),
    (" rn ", "\U0001f11c\u24a8"),
)
_CONFUSABLE_PROTOTYPES = {
    character: prototype for prototype, characters in _CONFUSABLE_PROTOTYPE_GROUPS for character in characters
}
if len(_CONFUSABLE_PROTOTYPES) != UNICODE_CONFUSABLES_SUBSET_SOURCE_COUNT:
    raise RuntimeError("the pinned Unicode confusable source table is incomplete or contains duplicate entries")


def _pinned_category(character: str) -> str:
    """Return a Unicode 15.1 category on every supported Python runtime."""
    codepoint = ord(character)
    # Unicode 15.1 added four ideographic-description symbols, one additional
    # symbol, and CJK Extension I. Python 3.12 ships UCD 15.0 and otherwise
    # reports these assigned characters as Cn.
    # Sources: https://www.unicode.org/Public/15.1.0/ucd/DerivedAge.txt and
    # https://www.unicode.org/Public/15.1.0/ucd/extracted/DerivedGeneralCategory.txt
    if 0x2EBF0 <= codepoint <= 0x2EE5D:
        return "Lo"
    if 0x2FFC <= codepoint <= 0x2FFF or codepoint == 0x31EF:
        return "So"
    return unicodedata.category(character)


def _safe_text(value: object) -> str:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value) if value.bit_length() <= 256 else ""
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return ""


def publication_semantic_text(value: object, *, strip_marks: bool = False) -> str:
    """Normalize public text and remove invisible identity-spoofing characters."""
    normalization_form = "NFKD" if strip_marks else "NFKC"
    text = unicodedata.normalize(normalization_form, _safe_text(value))
    semantic: list[str] = []
    for character in text:
        category = _pinned_category(character)
        if category[0] == "C" or character in _INVISIBLE_IDENTITY_CHARACTERS or (strip_marks and category[0] == "M"):
            continue
        if strip_marks and (category == "Pd" or character in {"\u2043", "\u2212"}):
            semantic.append("-")
        else:
            semantic.append(_SECURITY_TEXT_CONFUSABLES.get(character, character) if strip_marks else character)
    return "".join(semantic)


def publication_confusable_skeleton(value: object) -> str:
    """Return the pinned UTS #39 skeleton subset needed by reserved identities."""
    # The embedded values already contain the effective result of the UTS #39
    # NFD-first generation algorithm. Look up the original code point before
    # host normalization so newer pinned sources survive an older runtime UCD.
    mapped = "".join(_CONFUSABLE_PROTOTYPES.get(character, character) for character in _safe_text(value))
    # Identity placeholders are case-insensitive. Fold after the pinned map,
    # then close over characters such as ASCII M whose folded form is itself a
    # source. Mapping first preserves Unicode 17 sources unknown to the host.
    folded = unicodedata.normalize("NFD", mapped).casefold()
    remapped = "".join(_CONFUSABLE_PROTOTYPES.get(character, character) for character in folded)
    return unicodedata.normalize("NFD", remapped).casefold()


def _identity_key(value: object) -> str:
    skeleton = publication_confusable_skeleton(value)
    security_text = publication_semantic_text(skeleton, strip_marks=True)
    words = "".join(character if _pinned_category(character)[0] in {"L", "N"} else " " for character in security_text)
    return " ".join(words.split()).casefold()


_RESERVED_IDENTITY_SKELETONS = frozenset(_identity_key(identity) for identity in _RESERVED_IDENTITIES)


def publication_identity_present(value: object) -> bool:
    """Return whether identity text records non-placeholder provenance."""
    if not isinstance(value, str):
        return False
    identity = _identity_key(value)
    return bool(identity and identity not in _RESERVED_IDENTITY_SKELETONS)


__all__ = [
    "UNICODE_CONFUSABLES_GENERATOR_UCD_VERSION",
    "UNICODE_CONFUSABLES_SOURCE_SHA256",
    "UNICODE_CONFUSABLES_SUBSET_SOURCE_COUNT",
    "UNICODE_CONFUSABLES_VERSION",
    "publication_confusable_skeleton",
    "publication_identity_present",
    "publication_semantic_text",
]
