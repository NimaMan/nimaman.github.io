---
title: "COLON Token Launch: Investigation of Coordinated Trading Patterns"
date: 2025-01-21
description: "Technical investigation into how 68 coordinated wallets captured initial liquidity after a privately submitted transaction"
tags: ["ethereum", "mev", "trading-patterns", "defi", "investigation"]
math: false
---

# COLON Token Launch: Investigation of Coordinated Trading Patterns

## Executive Summary

On June 21, 2025, at Ethereum block 22750445, the COLON token launched with highly suspicious trading patterns. **68 brand-new wallets**, all funded exactly 33 hours in advance, somehow knew precisely when to submit transactions immediately after a **privately-submitted** "open trading" transaction, capturing the vast majority of initial liquidity.

The key question: **How did 68 different wallets know when to trade if the trigger transaction was not visible in the public mempool?**

## The Evidence

### 1. The Private Transaction

The "open trading" transaction (`0x6468b5e87161d9631f081f5f79d671baeaa77fef68787275ce97c5a3b51fed3a`) shows **no mempool confirmation time on Etherscan**. 

Compare:
- **Normal transaction**: Shows "Confirmed within 11 secs" ✅
- **COLON open trading**: Shows no confirmation time ❌

This definitively proves the transaction was submitted privately (via Flashbots, direct to builder, etc.) and was **not visible in the public mempool**.

### 2. The Statistical Impossibility

Yet somehow, 68 wallets achieved perfect coordination:

| Metric | Value | Probability if Organic |
|--------|-------|------------------------|
| New wallets (nonce 0) | 68/68 (100%) | < 0.000000001% |
| Identical gas fee | 5.000083545 Gwei | Impossible without automation |
| Pre-funded timing | All at block 22740446 | Impossible without coordination |
| Sequential positions | Captured slots 1-68 | Impossible without insider knowledge |

### 3. The Timeline

```
Block 22740446: All 68 wallets receive funding (~33 hours before)
Block 22750347: COLON token deployed (98 blocks before)
Block 22750445: Open trading + 68 coordinated buys
```

### 4. The Builder Connection

- **Block Builder**: BuilderNet (Beaver)
- **Validator**: `0xdadB0d80178819F2319190D340ce9A924f783711`
- **Result**: 68 out of first 69 transactions went to Banana Gun Router

## Technical Analysis

### How Private Transactions Work

When a transaction is submitted privately:
1. It bypasses the public mempool
2. Goes directly to specific builders/validators
3. Other users cannot see it before inclusion
4. Etherscan shows no "Confirmed within X secs"

### The Impossible Coordination

For 68 wallets to land perfectly after a private transaction requires:
1. **Advance knowledge** of when it would execute
2. **Perfect timing** of submissions
3. **Builder cooperation** for ordering

This cannot happen organically.

### Wallet Analysis

All 68 Banana Gun wallets shared these characteristics:
- First transaction ever (nonce 0)
- Funded at exactly block 22740446
- Amounts ranging from 0.040-0.055 ETH
- Buy amounts increasing linearly (algorithmic distribution)

### Verify on DexScreener

You can verify these transactions on DexScreener with filtered views:

**First 5 Buyers:**
1. [Wallet 1: 0xD9c4455587824C6B7ca8d3d35B64cf3b771863A4](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0?maker=0xD9c4455587824C6B7ca8d3d35B64cf3b771863A4)
2. [Wallet 2: 0x0e453430401980B7EfCCB1363e29cc62A923f538](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0?maker=0x0e453430401980B7EfCCB1363e29cc62A923f538)
3. [Wallet 3: 0x526a5b4C29c213f21c2F61De33Cb8E5395091bE3](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0?maker=0x526a5b4C29c213f21c2F61De33Cb8E5395091bE3)
4. [Wallet 4: 0x4e98d9f5Fbf2D633A6b21Ec404ee3870dE256472](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0?maker=0x4e98d9f5Fbf2D633A6b21Ec404ee3870dE256472)
5. [Wallet 5: 0xf53D0E6B502AB89071E1F967f60c4C920ce827B4](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0?maker=0xf53D0E6B502AB89071E1F967f60c4C920ce827B4)

[View full COLON/WETH pool on DexScreener](https://dexscreener.com/ethereum/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0)

## The Key Questions

### For BuilderNet (@BuilderNet_xyz)

1. How did 68 wallets know when to submit if the trigger was private?
2. Do you provide transaction visibility before block inclusion?
3. Was there an arrangement for this block?
4. Is this standard practice?

### For Banana Gun (@BananaGunBot)

1. How did you time 68 transactions after a private submission?
2. Why were all wallets brand new and pre-funded?
3. Do you have arrangements with builders?

## Implications

This pattern suggests:
- **Information Asymmetry**: Some actors have access to private transaction data
- **Coordinated Activity**: Organized buying to capture initial liquidity
- **Unfair Advantage**: Retail investors cannot compete with private information access

## Blockchain Evidence

All evidence is permanently recorded on Ethereum:
- Block: [22750445](https://etherscan.io/block/22750445)
- Token: [`0xD09ef011CE609AC504defCCB0B79A699184D0Bd4`](https://etherscan.io/token/0xD09ef011CE609AC504defCCB0B79A699184D0Bd4)
- Open Trading TX: [`0x6468b5e8...`](https://etherscan.io/tx/0x6468b5e87161d9631f081f5f79d671baeaa77fef68787275ce97c5a3b51fed3a)
- Pool: [`0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0`](https://etherscan.io/address/0xe9ffdb9ce7f2ffe214c0fe7fd0a73b5c2c8af9c0)
- Builder: BuilderNet (Beaver)
- Total suspicious transactions: 68

## Current Status

**January 21, 2025**: Awaiting response from BuilderNet regarding the technical mechanism that allowed this coordination.

## How You Can Verify

1. Check the open trading transaction on Etherscan - note the lack of mempool time
2. Examine block 22750445 - count the Banana Gun transactions
3. Check each wallet - verify they're all nonce 0
4. Calculate the probability of this happening organically

## Conclusion

The evidence reveals highly coordinated trading activity enabled by apparent access to private transaction information. The statistical impossibility of 68 organic users all creating new wallets, funding them 33 hours in advance, and submitting with identical gas parameters at the perfect moment raises serious questions about market fairness.

This investigation aims to increase transparency in the Ethereum ecosystem and understand how such coordination is technically possible.

---

*Last updated: January 21, 2025*

*For questions or additional information: nima.manaf8@gmail.com*