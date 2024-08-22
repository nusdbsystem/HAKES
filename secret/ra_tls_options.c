#include "attester.h" 

struct ra_tls_options my_ra_tls_options = { 

// SPID format is 32 hex-character string, e.g., 0123456789abcdef0123456789abcdef 

// replace below with your SPID
.spid = {{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,}}, 

.quote_type = SGX_LINKABLE_SIGNATURE, 

.ias_server = "api.trustedservices.intel.com/sgx/dev", 

// EPID_SUBSCRIPTION_KEY format is "012345679abcdef012345679abcdef01" 

// replace below with your subscription key 
.subscription_key = "012345679abcdef012345679abcdef01" 

}; 
