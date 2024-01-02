#ifndef MATRICES_H
#define MATRICES_H

#include <include/gemmini_params.h>

static const elem_t matrix_a[100][4][4] = {
{
    {0x3eee, 0xbf87, 0x3f58, 0xbe68},
    {0xbf3f, 0xbea9, 0xbe0b, 0x3f79},
    {0xbfb5, 0xbe85, 0xbf4a, 0x3f3a},
    {0xbf30, 0x3feb, 0xbceb, 0x3fad},
},
{
    {0x3f98, 0x3db8, 0x3dcb, 0x3f02},
    {0xbe2b, 0xbf2f, 0xbf68, 0xbe62},
    {0xbf3c, 0xbf43, 0x3f12, 0xbfc8},
    {0x3fdc, 0x3fec, 0xbf81, 0xbf31},
},
{
    {0x3e5a, 0xbf5f, 0x3f76, 0xbe1f},
    {0xbd2a, 0xbf31, 0xbf11, 0xbe5b},
    {0x3f35, 0x3e57, 0xbf70, 0xbf5d},
    {0x3f61, 0xbd4d, 0x3f31, 0x3fa5},
},
{
    {0x3e95, 0xbe81, 0xbda9, 0xbf8b},
    {0x3eee, 0x3d63, 0x3f22, 0xbe3e},
    {0x3f3e, 0x3fed, 0x3fe1, 0xbece},
    {0x3f64, 0x3f84, 0xbf4f, 0xbe9a},
},
{
    {0xbf70, 0xc003, 0xbea9, 0xbf14},
    {0x3eac, 0x3f2d, 0xbfc3, 0x3ff9},
    {0xbf1f, 0x3eaa, 0xbe53, 0xbf86},
    {0x3f7d, 0x3fd8, 0xbf8c, 0x3e39},
},
{
    {0x3ff6, 0xbfb0, 0x3f75, 0x3e18},
    {0x3ff6, 0x3f56, 0x3f77, 0x3f9c},
    {0xbf96, 0xbdf8, 0x3ffa, 0xbf30},
    {0xbe77, 0xbea1, 0xbf20, 0xbe7f},
},
{
    {0xbe6a, 0x3f87, 0x3f71, 0x3f41},
    {0x3f4b, 0x3de3, 0xbfab, 0xbf11},
    {0xbfab, 0xbe9a, 0xbea6, 0x3f9f},
    {0xbf5b, 0x3ff2, 0x3d82, 0xbf58},
},
{
    {0x401d, 0x3e34, 0x3f36, 0x3fa0},
    {0xbea5, 0x3f41, 0xbfd9, 0x3fcb},
    {0xbf23, 0xbe3a, 0xbf08, 0x3f03},
    {0xbe57, 0x3f42, 0xbfb1, 0x3e8b},
},
{
    {0x4001, 0xbe94, 0x3ffd, 0xbf8d},
    {0xbef7, 0xbe6c, 0x3fb5, 0x3fb3},
    {0xbf83, 0xbf27, 0x3daf, 0x3e46},
    {0x3fa2, 0xbf74, 0x3e91, 0xbfe6},
},
{
    {0xbeb3, 0xbf9d, 0xbfda, 0x3d93},
    {0x3f2b, 0xbf47, 0xbf99, 0x3ef2},
    {0xbf23, 0x3f75, 0x3d02, 0xbf18},
    {0xbf89, 0xbfcf, 0xbf56, 0xbf89},
},
{
    {0x3d11, 0x3ed3, 0xbd74, 0xbf9b},
    {0x3e3b, 0x3fac, 0xbd39, 0xbf45},
    {0xbe0d, 0xbf59, 0x3ed4, 0xbe33},
    {0xbf98, 0xbf97, 0xbe2b, 0xbdb0},
},
{
    {0x3dd4, 0xbd18, 0xbf2c, 0x3dca},
    {0xbe34, 0x3e0a, 0x3f9f, 0x3fcf},
    {0xbf6d, 0xbdf9, 0xbf8b, 0x3d94},
    {0x3fe7, 0xbf92, 0x3c86, 0xbf82},
},
{
    {0x3e8b, 0xbef7, 0x3f82, 0xbff8},
    {0xbe97, 0xbffe, 0x3f16, 0x3ece},
    {0x3d78, 0x3f4c, 0x3f87, 0xbf53},
    {0xbf53, 0x3e35, 0x3e9e, 0xbdd6},
},
{
    {0x3ef8, 0xbf9f, 0x3c7f, 0x3ee1},
    {0x3e42, 0xbf6a, 0x3f90, 0x3da5},
    {0x3e89, 0xbe8e, 0xbf66, 0xbfa2},
    {0xbef8, 0xbdb7, 0x3fdf, 0xbe88},
},
{
    {0x3f39, 0x3f79, 0x3ef5, 0x3ea2},
    {0x3f94, 0x3fe0, 0xbf13, 0xbf60},
    {0xbf54, 0xbfa7, 0x3f86, 0xbf90},
    {0x3f25, 0x3f11, 0x3d57, 0x3ef2},
},
{
    {0x3fc8, 0x3db2, 0x3fe1, 0xbc09},
    {0xbd4b, 0xbe31, 0x3ee7, 0x3f2d},
    {0xbe84, 0xbfa7, 0xbf51, 0xbf82},
    {0x3f91, 0x3f16, 0x3f10, 0xbf90},
},
{
    {0xbe85, 0xbf0f, 0x3fa5, 0xbf18},
    {0xbf0d, 0xc011, 0xbfa0, 0x3fe8},
    {0xbf14, 0x3ec7, 0xbf3e, 0x3ed5},
    {0xbd98, 0xbe81, 0x3f88, 0xbed2},
},
{
    {0xbefd, 0xbf80, 0x3edc, 0x3f80},
    {0x3ebb, 0x3f84, 0xbf39, 0xbf4e},
    {0x3f95, 0xbea5, 0x3c45, 0xbf84},
    {0x3f61, 0x3f91, 0x3d6d, 0xbcb4},
},
{
    {0x3e83, 0x3edf, 0xbf88, 0xbe50},
    {0x3faa, 0x3f71, 0xbf0b, 0x3f97},
    {0x3fac, 0x3f36, 0xbe06, 0x3f2e},
    {0x3ec2, 0xbf12, 0x3faf, 0x400f},
},
{
    {0xbe2d, 0x3e7b, 0x3f4a, 0x3f0b},
    {0xbf20, 0xbec8, 0x3d8a, 0x3dd3},
    {0xbde7, 0x3f21, 0x3f9f, 0x3f66},
    {0x3fc9, 0xbfa3, 0xbf20, 0x3edf},
},
{
    {0xbfc9, 0xbe88, 0xbf48, 0xbf34},
    {0x3f72, 0x3fb4, 0xbc1d, 0xbf93},
    {0xbea6, 0xbd29, 0xc00a, 0xbfd3},
    {0xbe85, 0x3f0f, 0xbeb6, 0xbfcd},
},
{
    {0xbf94, 0x3eb6, 0x3e96, 0xbf01},
    {0xbf48, 0xc00c, 0x3fc1, 0xbde4},
    {0x3f74, 0x3ee5, 0xbfb6, 0x3f8e},
    {0xbf01, 0xbf26, 0x3f7d, 0xbebe},
},
{
    {0xbfcd, 0xbe19, 0x3fa0, 0xbead},
    {0x3fc1, 0x3ee4, 0x3f8a, 0xbfc9},
    {0xbfd1, 0x3e90, 0xbf23, 0xbd8e},
    {0x3f8e, 0xc036, 0x3ea9, 0xbf86},
},
{
    {0x3f30, 0x3e19, 0xbe9e, 0x3f15},
    {0x3ec5, 0xbfe1, 0xbe3b, 0xbfcf},
    {0xbfac, 0x3f98, 0xbf03, 0xbdab},
    {0x3da6, 0xbfdd, 0xbffe, 0x3f22},
},
{
    {0x3f28, 0xbf66, 0xbf8d, 0x3e94},
    {0xbec3, 0xbfca, 0xbf12, 0xbc97},
    {0xbf53, 0xbf1d, 0x3fe4, 0x3d98},
    {0x3f93, 0xbf4c, 0x3ee5, 0xbf57},
},
{
    {0xbf94, 0xbeeb, 0xbdef, 0xbf75},
    {0xc004, 0xbde8, 0x3f69, 0x3fc4},
    {0x3c9e, 0x3e98, 0xbf00, 0x3e51},
    {0x3f3d, 0x400c, 0x3f22, 0xbef2},
},
{
    {0xbd23, 0x3e46, 0x3f1d, 0xbc87},
    {0xbfa1, 0xbf87, 0x3e48, 0xbe4b},
    {0x3ec9, 0x3fa0, 0xbec1, 0xbf2f},
    {0xc001, 0x3f83, 0x3f9c, 0xbeaf},
},
{
    {0xbe9c, 0x3eb3, 0x3f74, 0xbef1},
    {0xbf38, 0x3fa4, 0xbf11, 0x3f53},
    {0xbf80, 0x3f27, 0x3f85, 0xbf84},
    {0xbd2f, 0xbeb7, 0x3f91, 0xbf40},
},
{
    {0x3fe9, 0x3f60, 0x3fe3, 0xbf92},
    {0x3e71, 0x3e46, 0x3f86, 0x3d70},
    {0x3cb8, 0x3f50, 0xbf32, 0xbebb},
    {0xbdcb, 0xbfb2, 0x3d33, 0x3fa7},
},
{
    {0xbf0a, 0xbfe7, 0xbe86, 0xbfd8},
    {0x3f2f, 0xbeda, 0x3eab, 0xbf2d},
    {0x3f69, 0x3f12, 0x3f5d, 0xbfcd},
    {0x3f1a, 0xbf40, 0x3e84, 0x3f52},
},
{
    {0xbea8, 0x3c64, 0xbfbc, 0x3dd7},
    {0xbe87, 0xbf49, 0xbf6d, 0x3fce},
    {0xbc5c, 0xbf3d, 0xbec2, 0x3e63},
    {0xbf15, 0xbfa4, 0x3e9c, 0x3e9a},
},
{
    {0xbf1d, 0xbf61, 0xbe1b, 0x3fd0},
    {0x3f9a, 0xbf0d, 0xbd20, 0x3f71},
    {0x3f06, 0xbf82, 0x3ec4, 0xbe41},
    {0xbd03, 0x3e5a, 0xbfb5, 0xbd26},
},
{
    {0xbe07, 0x3eec, 0xbe63, 0x3f5f},
    {0x3fcb, 0x3f56, 0xbf23, 0xc00a},
    {0x3db6, 0xbe05, 0x3e25, 0xbf55},
    {0x3f6b, 0xbfbd, 0xbf3e, 0x3f3a},
},
{
    {0xbfcf, 0xbe7e, 0xbfa7, 0x3f36},
    {0x3f9b, 0x3e60, 0x3fc6, 0x3fd9},
    {0x3f8b, 0xbe76, 0xbf44, 0x3f21},
    {0xbf57, 0xbef0, 0xbf2a, 0x3f4c},
},
{
    {0x3f40, 0xc004, 0xbf13, 0x3e0c},
    {0xbf6b, 0xbe18, 0x3eef, 0xbf73},
    {0xbfeb, 0xbeab, 0x3fdc, 0xbee9},
    {0xbeb3, 0x3dc1, 0x3f1a, 0xbee4},
},
{
    {0xbdd4, 0x3dc0, 0x3fdc, 0x3f95},
    {0x3f52, 0x400d, 0xbe24, 0xc00c},
    {0xbf03, 0x3fc6, 0x3fe0, 0xbec4},
    {0xbfa3, 0xbfa7, 0xc01b, 0x3d9e},
},
{
    {0x3fcc, 0x3f19, 0xbe49, 0x3f94},
    {0xbf5a, 0x3f55, 0x3f15, 0xbfb2},
    {0x3e36, 0x3f2a, 0x3fd5, 0x3eaf},
    {0x3f83, 0x3e10, 0xbd8f, 0x3f86},
},
{
    {0x401b, 0xbfa9, 0x3d5a, 0x3e4b},
    {0xbf93, 0xbcfe, 0x3f8b, 0xbeeb},
    {0xbec4, 0xbf02, 0x3eae, 0xbf78},
    {0xbeb7, 0xbf56, 0x3f45, 0x3f61},
},
{
    {0x3b17, 0xbf9e, 0x3e82, 0xc015},
    {0xbf7f, 0xbfa9, 0x3f8d, 0x3ea2},
    {0xbf35, 0x3ec9, 0xbde7, 0xbf2e},
    {0x3ed5, 0x3d31, 0xbdfd, 0xbe56},
},
{
    {0xbfa3, 0x3ece, 0x3e34, 0xbf47},
    {0x3f51, 0xbec7, 0xbf04, 0xbd00},
    {0xbe9a, 0x3db9, 0xbfb4, 0x3e6c},
    {0x3f8e, 0x3ef8, 0x3f3a, 0xbf26},
},
{
    {0x3f23, 0x3eff, 0x3ee1, 0x3dba},
    {0xbf83, 0xbe54, 0x3e1e, 0xbf9f},
    {0x3e45, 0xbfaa, 0xbf21, 0xbf4c},
    {0xbfb7, 0xbfaf, 0x3f92, 0xbf82},
},
{
    {0x401e, 0x4014, 0xc009, 0xbea3},
    {0x3fe8, 0x3f9b, 0x3ff0, 0xbef1},
    {0x3fb7, 0x3f9a, 0xbc81, 0xbd1e},
    {0xbf2b, 0x3f86, 0x3e9f, 0x3f2d},
},
{
    {0xbe1a, 0xbf12, 0x3f8f, 0xbf33},
    {0x3e91, 0xc005, 0x3eee, 0xbefd},
    {0xbec3, 0xbf18, 0xbfaf, 0x3f98},
    {0x3f62, 0xbf92, 0x3e95, 0xbf88},
},
{
    {0xbeb0, 0xbf67, 0xc041, 0x3e94},
    {0x3ec5, 0x3f0e, 0x3ba5, 0x3e39},
    {0xbf98, 0xbe0e, 0xbe77, 0xbe09},
    {0x3fb2, 0x3f36, 0x3f21, 0x3f7a},
},
{
    {0x3f93, 0x3f79, 0xbe35, 0x3dba},
    {0x3f07, 0xbf99, 0xbe96, 0x3f8d},
    {0x3eb4, 0xbe0e, 0x3e56, 0x3fc3},
    {0xbfdb, 0x3e3a, 0xc018, 0xbe59},
},
{
    {0x3cc0, 0xbe4a, 0xbfa5, 0x3fd9},
    {0x3eb5, 0x3e6f, 0x3eff, 0xbf2d},
    {0xbfaa, 0xbf3d, 0xbff5, 0xbdb0},
    {0x3bd5, 0x3f40, 0x3ff7, 0xbd4d},
},
{
    {0xbf3c, 0xbf83, 0x3f46, 0x3f89},
    {0xc025, 0xbd0a, 0x3fa7, 0xbe85},
    {0x3f56, 0x3f40, 0x3f5e, 0xbf57},
    {0x3eed, 0x3f18, 0xbfa0, 0xbf8e},
},
{
    {0x3e9e, 0xc004, 0x3f45, 0x3f8c},
    {0x3d93, 0x3f90, 0x3e54, 0xbe79},
    {0xbf24, 0x3eca, 0xbee3, 0xbf4b},
    {0x3e77, 0x3ffa, 0x3f0e, 0xbf11},
},
{
    {0xbf31, 0x3ec7, 0x3fb9, 0x3e84},
    {0xbfd2, 0x3f86, 0xbea3, 0xbf62},
    {0xbf58, 0xbf63, 0x3ea5, 0xbfa2},
    {0xbf34, 0xbff1, 0x3f99, 0xbe83},
},
{
    {0xbf3e, 0xbdcc, 0x3ed4, 0xbfc3},
    {0x3f1e, 0x3fdb, 0x3e5b, 0xbf22},
    {0x3e99, 0x3eb7, 0xbf98, 0xc00f},
    {0x3edc, 0x3f7a, 0xbe69, 0xbf72},
},
{
    {0x3f3b, 0xbf90, 0x3eba, 0xbf34},
    {0x3f4c, 0xbea7, 0xbeb9, 0xbe44},
    {0x3b45, 0x3ef3, 0x3f92, 0xbeae},
    {0xc00a, 0xbf95, 0xbe3b, 0x3f9d},
},
{
    {0x3fd2, 0xbeb4, 0x3f41, 0xbf34},
    {0x3f64, 0x4014, 0xbd93, 0xbf88},
    {0x3f89, 0xbf29, 0xbf74, 0x3fb2},
    {0xbe91, 0xbed5, 0xbf21, 0x3fe0},
},
{
    {0xbe37, 0xbf84, 0x3f1b, 0x3fba},
    {0xbe12, 0xbf17, 0xbf44, 0x3ef5},
    {0x3f89, 0xbeb7, 0x3f28, 0xbf47},
    {0x3df2, 0x3f3a, 0x3da2, 0x3e80},
},
{
    {0x3e83, 0x3f0f, 0x3f55, 0x3fbf},
    {0x3f4e, 0x400a, 0xbe4d, 0x3ed6},
    {0x3f0b, 0x3e38, 0x3f98, 0x3ea5},
    {0xbec1, 0xbef6, 0x3fa1, 0xbdcb},
},
{
    {0x3ef5, 0x3f6f, 0x3fd6, 0x3f59},
    {0xbfc1, 0xbfd7, 0xbe2e, 0xbeec},
    {0x3e09, 0xbf0d, 0x3eb0, 0x3f20},
    {0x3faa, 0xbf3e, 0xbf82, 0xbe8a},
},
{
    {0x3e56, 0xbed6, 0xbd54, 0xbeb0},
    {0xbf95, 0x3f15, 0xbd45, 0x3fd3},
    {0x3e04, 0x3ec3, 0xbf85, 0x3ed6},
    {0x3f81, 0x3f6d, 0xbe59, 0xbe03},
},
{
    {0xbf16, 0xbe62, 0x403e, 0x3e8a},
    {0xbe31, 0x3fa4, 0xbf8c, 0xbe34},
    {0xbfe7, 0xbe73, 0xc009, 0xbf5e},
    {0x3f34, 0x3fbf, 0xbe1d, 0x3f54},
},
{
    {0x3ca8, 0xbdea, 0xbf5d, 0x3fa2},
    {0x3f04, 0x3fbe, 0xbf5b, 0xbdab},
    {0xbf95, 0xbf1f, 0xbe96, 0xbf87},
    {0xbfc2, 0x3e6a, 0x3f37, 0xbd90},
},
{
    {0x3f4d, 0x3ef6, 0xbf51, 0x3fd7},
    {0x3f9c, 0xbf48, 0xbfd7, 0xbd66},
    {0xbef1, 0xbeaa, 0x3f48, 0x3f82},
    {0xbf87, 0x3f87, 0xbe80, 0x3f4d},
},
{
    {0x3f2b, 0xbf2a, 0x400b, 0xbf0f},
    {0xc01e, 0x400d, 0xbf2c, 0xbead},
    {0x3fbb, 0x3edd, 0xbf89, 0xbf9c},
    {0x3ee6, 0xbf56, 0x3fa0, 0x3f3b},
},
{
    {0xbd79, 0xbf3b, 0x3fab, 0xbccd},
    {0xbef9, 0xc000, 0xbf6f, 0x3ead},
    {0x3e9e, 0x3f8a, 0xbf67, 0xbf00},
    {0x3e23, 0x3e95, 0x3ec7, 0xbecf},
},
{
    {0xbfd7, 0xbe57, 0xbf2e, 0xbedd},
    {0x3e08, 0x3e57, 0x3ef2, 0xc004},
    {0x3f93, 0xbf6e, 0xbde1, 0x3fb4},
    {0x3e08, 0xbe47, 0x3ed1, 0x3f7c},
},
{
    {0xbeae, 0xbe75, 0xbec6, 0xbee2},
    {0xbf2f, 0x3e7a, 0xbfbd, 0xc00f},
    {0xbf03, 0xbf6b, 0x3f0a, 0x3f27},
    {0xbe23, 0xbe89, 0x3eec, 0xbf73},
},
{
    {0x3fa8, 0xbff7, 0x3fe3, 0xbfe7},
    {0x3fa4, 0x3f7d, 0x3fd0, 0x3e02},
    {0x3f9a, 0x3e64, 0x3fab, 0x3f86},
    {0xbc06, 0x3f0e, 0xbe1c, 0xbf0f},
},
{
    {0x3ea9, 0xbd81, 0x3f35, 0x3f75},
    {0x3e3c, 0x3f80, 0xbf82, 0xbf02},
    {0xbf15, 0xbe30, 0xbe82, 0xbf44},
    {0x3f6e, 0xbfb2, 0x3fa6, 0x3faa},
},
{
    {0xbe6b, 0x3fe5, 0xbe85, 0x3ef9},
    {0xbedb, 0x4013, 0x3f97, 0x3f80},
    {0x3f6b, 0xbe48, 0xbfe8, 0xbf77},
    {0x403e, 0x3f85, 0x4008, 0xbed4},
},
{
    {0x3f30, 0xbf1a, 0x3f84, 0xbf3f},
    {0x3f34, 0xbf5b, 0x3f15, 0xbf51},
    {0xbf87, 0xbda7, 0xbd41, 0x3fdf},
    {0xbe3b, 0x3e2d, 0xbfab, 0x3e91},
},
{
    {0x3f98, 0x3fa4, 0x3ec4, 0xbf6b},
    {0xbfa7, 0xbf32, 0xbfa7, 0x3ec2},
    {0x3eb9, 0xbf9f, 0x3ee5, 0xbdf9},
    {0x3f9a, 0xbe44, 0xbdda, 0xbf85},
},
{
    {0x3ef4, 0x3deb, 0x3fc2, 0x4030},
    {0xc001, 0xbe2e, 0xbf9d, 0x3f6a},
    {0xbfae, 0xbec2, 0xbe33, 0x4018},
    {0xbf8b, 0xbf36, 0x3edd, 0xbe56},
},
{
    {0xbf17, 0xbf04, 0xbc4c, 0x3e3f},
    {0xbf21, 0x3fab, 0xbf9a, 0x400a},
    {0x3fec, 0xbe8d, 0x3fc6, 0x3f11},
    {0xc003, 0x3f9c, 0x3f63, 0x3f79},
},
{
    {0x3f92, 0xbf40, 0x3e99, 0xbe9b},
    {0x3f42, 0x3e84, 0xbf9c, 0x3f4e},
    {0xbd37, 0x3ea8, 0x3fdf, 0x3ec1},
    {0xbd31, 0xbf78, 0xbf55, 0x3f7a},
},
{
    {0xbe6b, 0xbe78, 0xbdd8, 0x3f53},
    {0x3ec2, 0x3f65, 0xbf16, 0xbede},
    {0xbedd, 0x3e5f, 0x3f2e, 0xbea4},
    {0xbf78, 0x3ffb, 0xbe09, 0x3f63},
},
{
    {0xbef0, 0x3f7c, 0x3f76, 0x3ecd},
    {0xbe63, 0x3ebc, 0xc035, 0xbeaf},
    {0xbed5, 0xbedd, 0xbf9b, 0x3f01},
    {0xc003, 0x3f8a, 0xbe1d, 0x3ee1},
},
{
    {0x3eee, 0x3e8f, 0x3fd2, 0x3f8a},
    {0x3dbb, 0xbf30, 0x3dab, 0xbfc4},
    {0xbe16, 0x3f90, 0xbe34, 0xbf63},
    {0xbff9, 0xbc82, 0x3f7a, 0x3f68},
},
{
    {0x4003, 0x3eb4, 0x3ee3, 0xbf9f},
    {0xbd49, 0x3ec7, 0x3eb8, 0xbfe0},
    {0xbf26, 0xbfa3, 0x3f60, 0xbf90},
    {0xbfd6, 0x3faf, 0xbf8c, 0x3fca},
},
{
    {0x3c82, 0x3fdb, 0x3e21, 0x3ed7},
    {0x3f5c, 0x3e84, 0x3f6c, 0xbe77},
    {0x3ea6, 0xbeb3, 0xbf60, 0x3ebc},
    {0xbf7f, 0xbed1, 0x3f19, 0x3e32},
},
{
    {0x3f71, 0xbf1d, 0xbfce, 0x3e50},
    {0xbf45, 0x3ef2, 0x3f91, 0x3f1e},
    {0x4029, 0x3ea6, 0x4026, 0xbfbc},
    {0x3ee6, 0xbef4, 0xbf7a, 0x3f84},
},
{
    {0xbe4f, 0xbf55, 0x3fcc, 0x3df5},
    {0x3f3b, 0xbfe0, 0x3ecf, 0x3fd9},
    {0xbfaf, 0xbf1b, 0xbf18, 0x3f83},
    {0x4003, 0x3e9c, 0xbf15, 0xbf21},
},
{
    {0x3ea5, 0xbdcd, 0x3f31, 0xbf95},
    {0xbe76, 0xbf82, 0x3ed7, 0x3e97},
    {0x3f37, 0xbf3c, 0x3fd7, 0xc008},
    {0x3f80, 0xbf6c, 0xbe82, 0xbfa3},
},
{
    {0xbff2, 0x3f59, 0x3f96, 0x3ec8},
    {0x3fb0, 0x3f90, 0x3f51, 0xbe95},
    {0x3f9c, 0xc015, 0x3eb9, 0xbe58},
    {0x3fa5, 0x3dce, 0x3f92, 0x3f44},
},
{
    {0x3ead, 0x3f5d, 0xbf6d, 0x3e36},
    {0xbdc0, 0x3fbc, 0x3fd1, 0x3f2f},
    {0xbfb0, 0xbf54, 0xbebf, 0xbd03},
    {0xbf15, 0xbf16, 0x3f9c, 0xbe66},
},
{
    {0xbf59, 0xbf84, 0xbf5e, 0x3e86},
    {0x3fa0, 0xbe93, 0x3e45, 0x3f98},
    {0xbf44, 0x3fb7, 0x3ef8, 0xbdf9},
    {0x3fce, 0xbebf, 0xbebd, 0x3f55},
},
{
    {0x3fa2, 0xbfad, 0x3fe7, 0x3ec9},
    {0xbe2b, 0xbe4d, 0x3f91, 0x3f8b},
    {0x3fb8, 0x3fca, 0x3f45, 0xbfed},
    {0xbe81, 0xbeb0, 0x3e16, 0xbf06},
},
{
    {0xbd16, 0xbf20, 0x3f09, 0x3e58},
    {0xbc57, 0x3f9c, 0xbe30, 0xbe3f},
    {0xbe59, 0xbe06, 0x3fd6, 0xbfd8},
    {0x3e33, 0x4017, 0xbec1, 0xbf16},
},
{
    {0xbe70, 0xbf36, 0x3cff, 0x3f25},
    {0xbfca, 0x4023, 0x3e1b, 0xbfcb},
    {0x3ed1, 0x3fab, 0xbdf0, 0x3f7b},
    {0xbe5e, 0x3e44, 0xbde8, 0xbfbc},
},
{
    {0xc002, 0x3e2f, 0xbd93, 0xbe2b},
    {0xbec0, 0x3fd3, 0x3f0d, 0xbedb},
    {0x3f17, 0x4002, 0x3edf, 0x3f53},
    {0x3fce, 0xbf17, 0x3f5e, 0x3f92},
},
{
    {0xbf16, 0xbff5, 0x3eeb, 0xbe7f},
    {0x3e64, 0xbe94, 0xbeeb, 0xbfe2},
    {0xbf16, 0x3ed1, 0xbfb5, 0xbe3a},
    {0x3ff4, 0xbf8c, 0xbfb5, 0xbe7c},
},
{
    {0x3fad, 0x3f97, 0x3f88, 0x3f35},
    {0x3e94, 0xbf36, 0x3f15, 0xbf77},
    {0xbf81, 0x400c, 0x3f3f, 0x3f73},
    {0xbf21, 0xbf72, 0xbfc7, 0xbc39},
},
{
    {0xbeba, 0x3f7c, 0x3e67, 0xbf4f},
    {0x3f10, 0x3fb6, 0x3f53, 0x3f2b},
    {0xbdca, 0xbfb0, 0xc016, 0x3fb7},
    {0x3e8f, 0xbe72, 0xbf8b, 0x3fab},
},
{
    {0x3f8d, 0xbfb3, 0xbef0, 0x3f5a},
    {0x4020, 0xbe2a, 0x400b, 0xbefd},
    {0x3fbf, 0x3f07, 0xbd0a, 0x3f5e},
    {0x3f69, 0x3dce, 0x3d2e, 0x3f72},
},
{
    {0xbf25, 0x400c, 0xbf58, 0xbefa},
    {0x3fc1, 0x3fb8, 0x3fe0, 0x403d},
    {0xbf59, 0x3f83, 0x3f1a, 0xbf2d},
    {0xbf3d, 0xc033, 0x3ebe, 0xbf96},
},
{
    {0x3f16, 0xbfbf, 0xbf45, 0x3e2b},
    {0xc027, 0xbf56, 0xbe51, 0xbea3},
    {0x3fda, 0x3f3a, 0x3f65, 0xbe27},
    {0x3f99, 0xbed2, 0x3e78, 0x3f62},
},
{
    {0xbd3e, 0x3ee0, 0xbf78, 0x3fb4},
    {0xbfef, 0x3f21, 0xc02c, 0x3ed1},
    {0xbd79, 0xbe74, 0xbe93, 0x3eee},
    {0x3eb4, 0x3e95, 0xbfa0, 0xbe1d},
},
{
    {0x3f1f, 0xbf5f, 0xbfc8, 0x3fe7},
    {0xbec5, 0x3f69, 0xbfd8, 0xbf05},
    {0xbce6, 0xbf3f, 0x3fc6, 0x3e42},
    {0x3c8f, 0x3f5d, 0x3fcd, 0xbf7b},
},
{
    {0xbd02, 0x3f97, 0x3f11, 0xbe48},
    {0xbeb0, 0xbf44, 0xbf1d, 0x3eaf},
    {0xbf76, 0x3f84, 0x3f86, 0xbfec},
    {0x3f83, 0xbf38, 0xbf1f, 0x3fe4},
},
{
    {0x3f4d, 0xbfc2, 0x3f49, 0x3e65},
    {0xc015, 0xbf96, 0xbe58, 0xbf50},
    {0xbd64, 0x3e82, 0xbf59, 0xbf84},
    {0x3f8a, 0x3fba, 0xbe2b, 0x3f9d},
},
{
    {0x3f16, 0x3ec1, 0xbf23, 0xbf7d},
    {0x3f05, 0x3d6c, 0xbf03, 0xbf32},
    {0x3e4e, 0xbf43, 0xbec8, 0x3f8f},
    {0xbef2, 0xbfb9, 0xbfb5, 0x3ffa},
},
{
    {0xbf47, 0xbec9, 0xbfbb, 0x3f01},
    {0xbeae, 0x4006, 0xbfe1, 0x3f9e},
    {0xbeca, 0xbf17, 0x3f14, 0xc025},
    {0xbf00, 0xbe95, 0x3def, 0x3ebd},
},
{
    {0xbfc5, 0x3f88, 0x3ff2, 0xbfcf},
    {0xbe86, 0x3f38, 0x3eeb, 0x3cb9},
    {0x3f3b, 0xbfaf, 0x3e9d, 0x3e0f},
    {0x4000, 0x3fdb, 0x3e95, 0x3f1b},
},
{
    {0xbfaa, 0xbe20, 0x3ccc, 0xbfaf},
    {0xbe7f, 0xbd82, 0xbeed, 0x3f1a},
    {0x3c87, 0xbd8e, 0xbee3, 0xbf77},
    {0x4011, 0x3eac, 0x3f45, 0xbf2a},
},
};

static const elem_t matrix_b[100][4][4] = {
{
    {0x3fba, 0xbf14, 0xbf2c, 0xbda0},
    {0xbf06, 0x3ed4, 0x3e5e, 0xbfba},
    {0x3d68, 0x3f26, 0xbe93, 0x3fda},
    {0xbf0c, 0x3e89, 0x4020, 0xbfd6},
},
{
    {0xbf6c, 0xbf89, 0xbf28, 0xbf7f},
    {0x3ee6, 0x400e, 0x3df8, 0xbf92},
    {0xbf41, 0x3e0a, 0x3fcb, 0xbf40},
    {0xbf9b, 0x3dbf, 0x3f2b, 0xbfe0},
},
{
    {0x3fb9, 0xbfb1, 0xbfb0, 0xbf8d},
    {0xbeaa, 0x4008, 0x3e6e, 0x3dc1},
    {0x3f58, 0x3f05, 0xbee2, 0xbfe3},
    {0xbf2e, 0xbef2, 0x3f64, 0xbf83},
},
{
    {0x3f35, 0x3eb1, 0x3f83, 0xbf85},
    {0x3f90, 0x3f7d, 0x3e3b, 0x3e8b},
    {0x3f90, 0x3b8c, 0xbeb1, 0xbf59},
    {0xbf6f, 0x400b, 0xbefd, 0xbeef},
},
{
    {0x3f91, 0x3e8a, 0x3f4c, 0xbea8},
    {0x3f5e, 0xc027, 0x3ed5, 0x3f70},
    {0x3f81, 0x3ecf, 0x3f6f, 0x3ece},
    {0x3dab, 0xbe5b, 0xbf55, 0x3f35},
},
{
    {0x3e50, 0x3e9f, 0xbf68, 0xbf1f},
    {0xbe2e, 0x3f13, 0xbb9b, 0x3c41},
    {0xbf74, 0xbece, 0x3ed9, 0x3efb},
    {0x3fe7, 0xbf98, 0x3e8e, 0x3ea1},
},
{
    {0xbebb, 0xbebf, 0xbed4, 0xbf23},
    {0xc013, 0x3efb, 0x3fd5, 0x3e3a},
    {0xc007, 0x3fd4, 0x3f9b, 0xbeb8},
    {0xbf33, 0x3e8b, 0x3f99, 0x3d97},
},
{
    {0xbf57, 0xbe98, 0x3f70, 0x3fe5},
    {0xbfb4, 0xbf86, 0xbf50, 0xbe89},
    {0x3fc5, 0xc008, 0x3ec6, 0x3e57},
    {0xbf08, 0x3f15, 0xc006, 0x3faf},
},
{
    {0xbed3, 0xbe87, 0x3f04, 0xbf91},
    {0x3e33, 0xbf98, 0x3fbd, 0xbfc6},
    {0xbff6, 0xbe06, 0x3f19, 0xbf7d},
    {0xbf6a, 0x3f5b, 0xbf62, 0xbef8},
},
{
    {0xbf2d, 0xbdb7, 0x3ff5, 0xbfb9},
    {0xbfe0, 0xbf55, 0x3f7c, 0xbf9e},
    {0xbfa7, 0xbfd1, 0xbe94, 0x3efa},
    {0x3f37, 0xbfb1, 0xbf4e, 0xbfe9},
},
{
    {0x3e59, 0x3f21, 0x3ffc, 0x3f52},
    {0x3f49, 0x3fab, 0x3e18, 0x3d6d},
    {0x3f9b, 0x3fca, 0x3f19, 0x3f91},
    {0xbf19, 0xbfa9, 0xbe9a, 0xbeab},
},
{
    {0x3eae, 0xbf5a, 0xbfa6, 0xbf95},
    {0xbf6f, 0xbe68, 0xbea5, 0xbf9f},
    {0xbfd8, 0xbf0f, 0x3e03, 0xbcfe},
    {0xbee3, 0x3eba, 0x3f28, 0x3ee6},
},
{
    {0xbdd2, 0xc024, 0xbfc5, 0xbf8f},
    {0xbf1f, 0xbf35, 0xbf0e, 0x400e},
    {0xc002, 0xbedf, 0x3efc, 0xbf7f},
    {0xbf22, 0x3edf, 0x3e46, 0xbf5c},
},
{
    {0xbe7e, 0x3f23, 0xbf6c, 0xbe89},
    {0x3f27, 0x3f87, 0xc012, 0x3f21},
    {0xbe6a, 0x3ed3, 0x3fc7, 0x3ded},
    {0x3f48, 0x3e12, 0x3f66, 0xbee7},
},
{
    {0x3faf, 0x3eb5, 0xbfbe, 0xbf15},
    {0x3f22, 0x3e07, 0x3eb2, 0x3f9d},
    {0x3fe2, 0xbf4f, 0xbfc9, 0x3f62},
    {0xbf95, 0xbe4d, 0x3f16, 0x3de4},
},
{
    {0x3f1b, 0x3eff, 0x3f86, 0xbd91},
    {0x405f, 0x3fb4, 0x3ea2, 0xbe72},
    {0xc019, 0x3ec1, 0xbf83, 0x3d78},
    {0xbfe8, 0xbf30, 0xbfc2, 0x3d50},
},
{
    {0xc02f, 0xbd45, 0xbf05, 0xbe4f},
    {0xbf8e, 0x3e7d, 0xbdb2, 0x3fe8},
    {0xbf89, 0xbeab, 0x3f2b, 0xbf2e},
    {0xbf96, 0xbfe8, 0x3f74, 0x3eb6},
},
{
    {0x3d75, 0xbf11, 0x3fb8, 0x3f1f},
    {0xbe92, 0xbf12, 0xbee2, 0x3f99},
    {0x3f7b, 0x3f9a, 0x3f57, 0x3f24},
    {0xbe9b, 0x3e92, 0xbfa1, 0xbe29},
},
{
    {0x3f17, 0x3fb2, 0xbe17, 0xbfc8},
    {0xbf26, 0x3f13, 0xbc99, 0xbe7f},
    {0xbebc, 0xbf05, 0x3e3a, 0xbd83},
    {0x3f76, 0xbf81, 0x3f05, 0xbe2c},
},
{
    {0xbd05, 0x4027, 0x3fc4, 0xbe57},
    {0xbf32, 0xbfef, 0x3edb, 0xbf8a},
    {0xbfc4, 0x3f85, 0xbf52, 0xbee4},
    {0xbf3d, 0xbfd9, 0xbf5d, 0xbfca},
},
{
    {0x3e6d, 0x3f39, 0xbe2c, 0xbdc5},
    {0xbf57, 0x3fe0, 0xbe7c, 0xbe4d},
    {0x3f65, 0xbf3c, 0x3e87, 0xbde7},
    {0xbf78, 0x3ef8, 0xc039, 0x3fd7},
},
{
    {0xbf67, 0x3e06, 0xbfa0, 0xbf11},
    {0xbf81, 0xbfa7, 0xbedc, 0x3f8b},
    {0x3fb3, 0x3f0b, 0x3ecb, 0x3ee5},
    {0xbe3d, 0xbea8, 0x3e97, 0xbe57},
},
{
    {0x4011, 0xbd14, 0xbf36, 0x3f4f},
    {0xbeb0, 0xbfbb, 0xbf6f, 0xbfbe},
    {0x3faf, 0xbf9c, 0x3f44, 0xbf07},
    {0xbf10, 0xbf0a, 0x3da8, 0x3d07},
},
{
    {0x3f88, 0xbfcc, 0xbf94, 0xbfa8},
    {0x3f66, 0x3fa9, 0x3fb1, 0x3ee2},
    {0xbe9a, 0x3f70, 0xbcb6, 0x3e74},
    {0xbfb1, 0x3f41, 0x3efc, 0x3ecf},
},
{
    {0xbf30, 0xbf8f, 0xbec5, 0x3f01},
    {0xbff2, 0xbf56, 0x3e12, 0xb6b8},
    {0x3fbd, 0x404c, 0xbf7a, 0x3f31},
    {0x3eeb, 0x3fbe, 0x3e4c, 0xc06f},
},
{
    {0xbea0, 0xbb58, 0x3eee, 0xbfb6},
    {0xbf02, 0xbfe8, 0xbe40, 0xc00a},
    {0x4015, 0x3e15, 0x3f60, 0xbe25},
    {0x3e1e, 0x3e45, 0x3faa, 0xbdaa},
},
{
    {0xbe2a, 0xbf37, 0xbfa7, 0x3f30},
    {0xbe20, 0x3e96, 0x3f79, 0xbddc},
    {0x3f66, 0x3e0c, 0xbe88, 0xbf04},
    {0xc000, 0xbe9e, 0xbca1, 0xc006},
},
{
    {0xbf24, 0xbede, 0xbf99, 0xc00e},
    {0xbe52, 0x3fba, 0x3f73, 0x3ecc},
    {0xbfce, 0x3f20, 0x3e23, 0x3ec0},
    {0x400e, 0xbd9a, 0x3e85, 0xbf17},
},
{
    {0x3f57, 0x3f3d, 0x3ef0, 0xbfa6},
    {0x4005, 0x3e4d, 0xbeaa, 0xbfd3},
    {0x3f96, 0x3ee6, 0x3f38, 0x3f29},
    {0xbd76, 0xbf87, 0x3ed8, 0xbfec},
},
{
    {0x3e27, 0xbf79, 0xbdeb, 0x3e23},
    {0x3e87, 0x3e06, 0x3ed1, 0x4023},
    {0xbf2c, 0x3e0f, 0xbf70, 0xbf00},
    {0x3d9f, 0xbee4, 0x3f2c, 0xbea2},
},
{
    {0xbf4d, 0xbd23, 0x3cd2, 0xbd81},
    {0xbe3a, 0x3fcd, 0xbf2f, 0xbd8d},
    {0xbb45, 0x3f99, 0xbfdf, 0x4006},
    {0x3d82, 0xbec4, 0x3f21, 0x3f0d},
},
{
    {0x4004, 0xbf1c, 0x3efc, 0xc013},
    {0xbe1d, 0x3f39, 0x3fcf, 0x3f85},
    {0x3f9d, 0x3fd5, 0xbf3b, 0xbed0},
    {0xbf9c, 0xbf2b, 0xc004, 0xbfa6},
},
{
    {0xbd98, 0x3f79, 0xbf25, 0x3fa1},
    {0x3f09, 0x3f99, 0xbe86, 0xbfd3},
    {0x3ed2, 0xbedf, 0xbf00, 0x3fb9},
    {0xbed6, 0xbfb3, 0xc006, 0x3ec5},
},
{
    {0x3ec4, 0x3f3f, 0xbfbf, 0xbf14},
    {0x3f95, 0xbf80, 0xbe9e, 0x3d59},
    {0x3fb4, 0x3f91, 0x3f8e, 0xbeaf},
    {0x3fa3, 0x3e4f, 0xbed2, 0xbf5b},
},
{
    {0xbf3c, 0x3df5, 0x3f4c, 0x3fba},
    {0xbcfe, 0x3e96, 0x3dd5, 0xbeb8},
    {0xba21, 0xbf36, 0xbf09, 0x3d97},
    {0x3d1c, 0xbfe8, 0x3fb0, 0xbf53},
},
{
    {0x3f40, 0xbeaa, 0x3fa3, 0x3e58},
    {0x3f98, 0xbf7c, 0x3e8a, 0x3f17},
    {0x3e76, 0x3fa3, 0xbf8d, 0xbf4d},
    {0xbe31, 0x3d84, 0x3f25, 0x3fab},
},
{
    {0xc006, 0xbf1a, 0x3f8b, 0x3eea},
    {0x3ee1, 0xbb91, 0x3e0d, 0x3dfb},
    {0x3f96, 0x3f15, 0xbfd3, 0xbfda},
    {0xbeda, 0x3d94, 0x3f90, 0xbf79},
},
{
    {0x3f8d, 0x3f10, 0xbe6c, 0x3ed9},
    {0xbe24, 0x3ed4, 0x3dc3, 0xc00e},
    {0xbea3, 0xbf8d, 0x3db6, 0xbf14},
    {0xbfcf, 0x3f8c, 0xbfc3, 0x3e97},
},
{
    {0xbe2e, 0x3f1e, 0xbebd, 0x3f74},
    {0x3f93, 0xbf89, 0x3d7d, 0xbf87},
    {0xbe83, 0xbf3b, 0x3f30, 0x3ff1},
    {0x3ef2, 0xbfe3, 0x3fc5, 0x3fb5},
},
{
    {0xbf6f, 0xbfc4, 0x3f23, 0x3fb0},
    {0x3efa, 0x3f9f, 0x3e7e, 0x3f12},
    {0xbf31, 0x3e98, 0x3f07, 0xbfa0},
    {0xbf6b, 0x3e2e, 0x3ded, 0x3cf1},
},
{
    {0xbf92, 0xbee0, 0x401b, 0xbf27},
    {0x3f4e, 0x3f9f, 0x3eb9, 0xbf01},
    {0x3f62, 0xbd65, 0xc00f, 0x3fc5},
    {0xbdbc, 0xbf0c, 0x400c, 0x3fa6},
},
{
    {0x3f75, 0x3f7f, 0xbedd, 0xbfed},
    {0xbf57, 0x3f18, 0x3e83, 0x3d96},
    {0x3e0f, 0x3eb4, 0xbe37, 0x3ebe},
    {0x3f51, 0x3f03, 0x3f03, 0xbf3e},
},
{
    {0x3f95, 0x3ffe, 0x3e04, 0xbe2b},
    {0x3ec1, 0x3def, 0xbeb9, 0x3f35},
    {0x3eba, 0x3d7b, 0xbf4c, 0xbf83},
    {0xbf88, 0x3f54, 0xbfec, 0x3faf},
},
{
    {0xbf24, 0x3f05, 0xbe53, 0x3cc5},
    {0x3f79, 0x3f5c, 0xbd9d, 0x3dcf},
    {0x3e9f, 0xbe4d, 0xc008, 0x3ff1},
    {0xbd9a, 0x3f92, 0x3f12, 0xbf94},
},
{
    {0x3f12, 0x3f2d, 0x3f9a, 0x3ebd},
    {0xc00a, 0x3ef2, 0xbf32, 0xbe1d},
    {0x3fb8, 0xbf0a, 0xbeb2, 0xbfa4},
    {0x3ee5, 0xbfbb, 0x3eb6, 0xbd33},
},
{
    {0xbf54, 0x3e3a, 0xbf32, 0xbf12},
    {0x3fb0, 0xbf07, 0x3e82, 0xbf5f},
    {0x3fb0, 0x3e29, 0xbf56, 0xbd07},
    {0xbe56, 0xbf40, 0xbfb4, 0x3f33},
},
{
    {0x3e68, 0x3f5b, 0x3eb3, 0xbf9b},
    {0xbf17, 0x3f23, 0x3d9a, 0xbdd0},
    {0xbefc, 0xbf80, 0xbf38, 0xbf78},
    {0xbfb3, 0x3e4a, 0x3d1d, 0xbf02},
},
{
    {0xbf21, 0x3fce, 0xbe2c, 0xbf0c},
    {0x3fe4, 0xbfb5, 0x3ed9, 0xbfd2},
    {0x3fa5, 0xbf94, 0xbfa0, 0x3e6b},
    {0x3f86, 0xc01b, 0x3efa, 0x3f25},
},
{
    {0xbeda, 0x3f83, 0xbdee, 0xbfb2},
    {0x3f88, 0x3f77, 0xbe6e, 0xbf85},
    {0x3d98, 0xbf5c, 0x3f94, 0xc013},
    {0x3f92, 0x3e20, 0x3f78, 0xbea1},
},
{
    {0x3fc7, 0x3fae, 0x3f32, 0x3e98},
    {0x3e00, 0xbdea, 0xbf1a, 0x3ffa},
    {0xbf7c, 0xbfdc, 0x3fcc, 0x3f5c},
    {0xbecb, 0x3f0b, 0xbed9, 0xbff3},
},
{
    {0xbf4b, 0xbe20, 0xbed9, 0x3fa2},
    {0xbee4, 0xbfff, 0x3ddd, 0xbf74},
    {0x3fc9, 0x3f5b, 0xbc2b, 0x3f6a},
    {0xbf55, 0x3e83, 0xc016, 0xbe8e},
},
{
    {0x3f62, 0x3e75, 0xbe7d, 0xbf50},
    {0xbe84, 0x3f2c, 0xbeb0, 0xbf61},
    {0xbf8f, 0x3f89, 0xbeb1, 0xbf5d},
    {0xbf98, 0xbf94, 0x3e98, 0x3f34},
},
{
    {0xbde5, 0xbf90, 0x3dd6, 0x3f83},
    {0x4004, 0x3db0, 0xbfbc, 0xbe1f},
    {0x3f1b, 0x3e55, 0xbf9c, 0x3e32},
    {0x3ff6, 0x3fd2, 0x3da0, 0x3cd8},
},
{
    {0xbf99, 0x3e4e, 0x3f90, 0xbee0},
    {0x3fc4, 0x3c31, 0x3f1c, 0xbfbf},
    {0x3fd0, 0xc005, 0x3e7b, 0x3f9e},
    {0x3f35, 0xbe8d, 0xc00a, 0xbf00},
},
{
    {0xbd85, 0xbe4b, 0xbffa, 0x3ede},
    {0xbf78, 0xbeec, 0x3ec5, 0xbeaa},
    {0x3f8c, 0xbe02, 0x3f9d, 0x3e9a},
    {0x3eed, 0x4015, 0xbf76, 0xbe86},
},
{
    {0x3f70, 0x3f63, 0xbf06, 0xbf93},
    {0xbf7f, 0xbfad, 0x3f96, 0xbf9f},
    {0xbf43, 0x3f4d, 0x3fd9, 0x3e8e},
    {0xbf68, 0xbfe9, 0xbf21, 0xbfdd},
},
{
    {0xbeb3, 0x3f6e, 0x3fc2, 0x3fb2},
    {0x3f39, 0x3ed0, 0xbff9, 0x3f92},
    {0x3df5, 0x3d47, 0x3e10, 0x3f81},
    {0xbf18, 0x3cf1, 0xbf70, 0xbece},
},
{
    {0xbf8b, 0xbf01, 0xbea8, 0x3eed},
    {0x3f83, 0x3fb1, 0xbfff, 0x3e55},
    {0x4016, 0xbf33, 0x3ec3, 0xbf86},
    {0xbf9c, 0xbfce, 0xbf42, 0x3fc5},
},
{
    {0x3f26, 0xbfe5, 0xc03b, 0xbf8b},
    {0xbe8f, 0x3f0d, 0xbfc3, 0xbfd5},
    {0x3f16, 0xbfed, 0xbf87, 0xbf3e},
    {0xc04a, 0xc018, 0xbf14, 0xbeb9},
},
{
    {0x3f33, 0xbf8b, 0xbf3f, 0x3f43},
    {0x3f9c, 0xbfc8, 0xbfc7, 0xbf1d},
    {0x3fa4, 0x3efc, 0xbf20, 0x3f4a},
    {0x3f66, 0xbfe0, 0xbeef, 0xbc64},
},
{
    {0x3f5f, 0x3e9d, 0xbc92, 0x3d66},
    {0x3f91, 0x3e8b, 0xbe97, 0xc000},
    {0x3f22, 0x3ffb, 0xbed4, 0xbe8c},
    {0xbf0d, 0x3f4a, 0xbfac, 0x3ea9},
},
{
    {0xbfed, 0xbd86, 0x3fae, 0xbfad},
    {0xbf70, 0xbd0a, 0xbe29, 0xbf54},
    {0x3e75, 0xbf60, 0x4019, 0xbf98},
    {0x3e5c, 0xbf11, 0xbf6a, 0x3f32},
},
{
    {0x3e6d, 0x3e0e, 0x3de1, 0xbf86},
    {0xbff7, 0x3fbe, 0xbf0b, 0x3fab},
    {0xbfcc, 0x3f12, 0xc00e, 0xbf6a},
    {0x3eb7, 0x3f87, 0x3f28, 0xbdc3},
},
{
    {0x4040, 0xbdf7, 0xbea6, 0x3ee3},
    {0x3f90, 0x3e7d, 0x3f4e, 0x3f63},
    {0x3e47, 0x3f0d, 0x3e9d, 0x3ee6},
    {0x3f89, 0xbef9, 0x3e08, 0xbe6a},
},
{
    {0xbf4e, 0xbf3a, 0x3dd8, 0x3f6e},
    {0xbe8b, 0xbf2f, 0x3faf, 0xbe1d},
    {0xbdde, 0x3e52, 0xbfac, 0xbe11},
    {0xbf36, 0xbfa0, 0xbe8f, 0x3e11},
},
{
    {0x3fa5, 0xbf98, 0xbf90, 0x3d68},
    {0x3e90, 0x3ee2, 0x3ff1, 0xbf4f},
    {0xbf32, 0x3ddd, 0x3fd5, 0x3f73},
    {0x4007, 0xbea0, 0xbd86, 0x3e34},
},
{
    {0x3ea9, 0x3f60, 0xbed4, 0x3fb1},
    {0xbfa9, 0x3db7, 0xbfca, 0xbd97},
    {0x3d72, 0x3d9b, 0x3f6e, 0x3f1b},
    {0xbf33, 0x3fa0, 0xbed9, 0x3ecb},
},
{
    {0x3e5c, 0x3f96, 0x3fa5, 0x3dca},
    {0x403c, 0x3f63, 0xbe17, 0x3e39},
    {0xbf68, 0xbf16, 0xbe6f, 0x3f6b},
    {0x3f79, 0xbe77, 0x3e05, 0x3f33},
},
{
    {0xbf3d, 0x3fbd, 0x3f82, 0xbfe8},
    {0x3f3c, 0x3e6f, 0x3e8f, 0x3f4a},
    {0x3f9b, 0xbc09, 0xbf67, 0xbd0f},
    {0x3e50, 0x3f0c, 0x3e4a, 0x3e6c},
},
{
    {0x3f8e, 0x3e10, 0x3e4e, 0xbd8f},
    {0xbee9, 0x3ece, 0xbfe6, 0xbf53},
    {0xbff4, 0xbf37, 0x3ee5, 0x3e1e},
    {0xbf07, 0x3f8a, 0x3f23, 0x3eac},
},
{
    {0x3dfc, 0x3ff0, 0x3f07, 0xbef1},
    {0x3f4f, 0xbf7a, 0x3f2a, 0xbf09},
    {0x3f2d, 0x3ee7, 0x3f5e, 0xbe9b},
    {0xbf51, 0xbf7c, 0xbefa, 0x3ff0},
},
{
    {0xbfa1, 0x3fee, 0xbeff, 0xbf44},
    {0x3f92, 0xbf21, 0x3e97, 0xbe1c},
    {0x3f5a, 0xbdb4, 0xbfb0, 0xbfca},
    {0xbecb, 0x3f3c, 0xbf12, 0xbf87},
},
{
    {0xbf07, 0xbf46, 0xbf0d, 0xbf7d},
    {0x3f1a, 0x4042, 0x3fcc, 0x3f6b},
    {0xbdab, 0x3e1d, 0x3ea2, 0x3f98},
    {0x3f03, 0xbf39, 0xbf1d, 0x3fe7},
},
{
    {0x3fa0, 0x3fa2, 0xbf87, 0xbf1a},
    {0x3f19, 0xbfd9, 0xbe3b, 0x3e1a},
    {0x4002, 0xbf6b, 0xbf26, 0x3eb0},
    {0xbe9e, 0xbf75, 0xbfd7, 0x3ec4},
},
{
    {0xbfec, 0xbf87, 0xbcb6, 0xbf8e},
    {0xbdd7, 0x3e92, 0x3f41, 0xbee6},
    {0xbf8a, 0x3f02, 0xbf73, 0x3ddf},
    {0x3ee6, 0x3d0c, 0xbec8, 0x3f71},
},
{
    {0xbef1, 0x3fba, 0xbf8f, 0x3fb9},
    {0x3f8e, 0x400b, 0xbf93, 0x3ee3},
    {0x3d86, 0xbe86, 0x3f99, 0x3eda},
    {0xbfd6, 0x3f3c, 0xbdf9, 0x3fb9},
},
{
    {0x3f2e, 0x3ca9, 0xbeca, 0xbeff},
    {0x3f97, 0x3e4f, 0xbfab, 0xc013},
    {0x3f33, 0x3f66, 0xbf79, 0xbf62},
    {0x3f5e, 0x4009, 0x3f9a, 0x3e15},
},
{
    {0x3fa6, 0xbef2, 0xbf2c, 0xbe77},
    {0xbe92, 0xbeb3, 0x3fee, 0x3fa7},
    {0x3fdc, 0x3f8c, 0xbeaa, 0xbf13},
    {0xbf5d, 0xbe7a, 0xbf68, 0xbed6},
},
{
    {0xbf0c, 0xbe94, 0x3ed5, 0x3f1c},
    {0x3e48, 0x3e20, 0x3f80, 0x3f19},
    {0x3f12, 0xbf41, 0xbed4, 0x3e5c},
    {0xbd8e, 0xbde1, 0xbf8e, 0xbe53},
},
{
    {0xbfb1, 0xbfc8, 0xbfd5, 0xbfad},
    {0xbcc3, 0xbf5f, 0x3e28, 0xbe0d},
    {0xbf4d, 0xbf17, 0x3fad, 0x3f94},
    {0xbd10, 0x3de7, 0x3ed9, 0x3f5d},
},
{
    {0x3f65, 0x3e40, 0x3f7d, 0xbdcc},
    {0xbec5, 0x3ef9, 0x3e6a, 0xbfc9},
    {0xbef1, 0xbe46, 0xbe8c, 0xbed6},
    {0x404f, 0x3c71, 0x3f23, 0xbec9},
},
{
    {0xbf62, 0x4023, 0x3f11, 0x3f24},
    {0x3fa0, 0xbfaa, 0x3f52, 0x3e0a},
    {0xbfaf, 0xbd5c, 0xbfa9, 0xbe29},
    {0x3ee8, 0x3f64, 0x3f0a, 0xbe12},
},
{
    {0x3ecf, 0x3f5f, 0x3f36, 0xbf20},
    {0x3f8e, 0xbf8c, 0xbf8d, 0xc009},
    {0xbf61, 0x3dc4, 0x3e8f, 0x3f43},
    {0xbdf7, 0x3f85, 0xbeb7, 0xbea5},
},
{
    {0x3f51, 0xbf6a, 0x3fc6, 0xbf8c},
    {0x3f53, 0x3eec, 0x3fa6, 0xbd1a},
    {0xbb45, 0x3f86, 0x3de9, 0x3eee},
    {0xbf9c, 0x3e1e, 0xc006, 0x3bed},
},
{
    {0xc045, 0xbee2, 0x3f51, 0x3dd0},
    {0x3d87, 0xbee3, 0x3d51, 0xbfb9},
    {0xbc06, 0x3e19, 0xbfc7, 0xbd00},
    {0x3e3c, 0xbee4, 0xbd5f, 0x3f05},
},
{
    {0xbf4d, 0x3d24, 0xbed4, 0x3ec1},
    {0xbf9a, 0x3ec0, 0xbe89, 0x3f4c},
    {0xbfea, 0xbf75, 0xbfa5, 0xbef1},
    {0x3f8c, 0xbfb1, 0xbf8f, 0x3fff},
},
{
    {0x3f6f, 0x3f9a, 0x3fac, 0x3fa8},
    {0x3e24, 0x3e8d, 0xbf1a, 0xbf4f},
    {0xc014, 0x3eaf, 0xbe0e, 0xbe11},
    {0x3f65, 0x3f7e, 0x4032, 0x3fc1},
},
{
    {0x3f3f, 0xbe70, 0xbf5f, 0x3e23},
    {0x3c97, 0x3e53, 0xbeae, 0x3f13},
    {0xbf78, 0x3f11, 0xbee2, 0x3fef},
    {0x3f2a, 0x3f4c, 0x3f7a, 0xbebd},
},
{
    {0xbe29, 0x3fa0, 0x3f4e, 0x3fba},
    {0x3f7e, 0xc00a, 0xbed8, 0x3f1a},
    {0x3fe4, 0xbf85, 0x3f92, 0xbff2},
    {0x3c4c, 0x3fcc, 0x3ee7, 0xbeb5},
},
{
    {0xbf48, 0xbf47, 0xbf7e, 0x3f9d},
    {0xbf8a, 0x3e0a, 0xbee9, 0x3f8a},
    {0xbf4d, 0xbe8d, 0x3f8f, 0xbfd8},
    {0x3ee0, 0x3f0f, 0xc015, 0x4003},
},
{
    {0x3ee8, 0x3f2a, 0x3ec5, 0xbd42},
    {0xbf31, 0x3fac, 0x3fde, 0xbf52},
    {0x3fb9, 0x3f5a, 0x4021, 0xbf7f},
    {0x3d18, 0x3e8d, 0xbee5, 0x3f19},
},
{
    {0x3f88, 0xbfdf, 0xbfa3, 0xbf05},
    {0x3f89, 0xbfd8, 0x3f82, 0x3e6e},
    {0x3d84, 0x3f31, 0x3efb, 0xbf87},
    {0x3f87, 0x3fba, 0x3e06, 0xbed4},
},
{
    {0x3fee, 0x4032, 0x3f20, 0x3f7c},
    {0x3dc4, 0x3e00, 0x3ee4, 0xbf83},
    {0xbe15, 0xbf88, 0xbe4d, 0xbf65},
    {0x3e33, 0x3e9f, 0x3e92, 0xbf5a},
},
{
    {0x3ea9, 0xbc89, 0x3ef2, 0x3f85},
    {0x3fb3, 0x4000, 0xbf38, 0x3e47},
    {0xbe5e, 0xbf25, 0x3ede, 0xbe9a},
    {0x3e5f, 0xbf2d, 0x3f25, 0xbf48},
},
{
    {0x3fa4, 0x3f17, 0xc00d, 0xbf83},
    {0x3e88, 0x3f7f, 0xbfd2, 0xbd19},
    {0x3ed0, 0x4003, 0x3fc3, 0xbfcf},
    {0x3fdb, 0x3f12, 0x3fe5, 0x3f32},
},
{
    {0xbfb0, 0x3e32, 0x3f35, 0x3ff5},
    {0x3e6b, 0xbfb4, 0x3f16, 0xbeaf},
    {0xbe9d, 0x3f8a, 0xc005, 0xbcd2},
    {0xbf31, 0xbfa1, 0x3f9a, 0x3f90},
},
{
    {0x3f2c, 0xbf1e, 0xbfb6, 0x3ef3},
    {0xbf1e, 0xbe3e, 0xbeb1, 0x3c74},
    {0xbe8e, 0x3e63, 0x3fb8, 0xbf90},
    {0x3e99, 0xbefb, 0xbea2, 0x3dfc},
},
{
    {0x3f8e, 0xbf7b, 0xbeb4, 0xbf0e},
    {0xbfec, 0x3fca, 0xbf9f, 0xc013},
    {0x3fdc, 0x3fb6, 0x3f21, 0x4008},
    {0x3f6b, 0xbf93, 0x3f8d, 0x3f51},
},
{
    {0xbf83, 0xbec9, 0x3f6c, 0xbedc},
    {0x3f35, 0x3f52, 0xbf08, 0x3f82},
    {0xbf0b, 0xbf49, 0xbf74, 0x3fa0},
    {0xbeb8, 0x3e79, 0x3dac, 0x3f24},
},
{
    {0xbd30, 0x3f80, 0xbeed, 0xbe9b},
    {0xbfbd, 0xbf5a, 0x3ee4, 0x3fb1},
    {0xbfbb, 0x3e71, 0xbfb0, 0xbe05},
    {0x3ed2, 0xbf6a, 0xbe28, 0x3f81},
},
};

#endif /* MATRICES_H */