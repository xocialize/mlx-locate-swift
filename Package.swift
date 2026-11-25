// swift-tools-version: 6.2
// mlx-locate-swift — text-to-boxes grounding on Apple silicon: a picture and a phrase in, boxes out
// (MES image.locate, AB-A-0293). Qwen3-VL-4B-Instruct (Apache-2.0) through mlx-swift-lm's MLXVLM.
//
//   Locate       engine-agnostic core: Locator.locate(image:phrase:) -> [LocatedBox], scored
//   MLXLocate    the MLXEngine package (imageLocate: typed boxes; imageAnalysis: `answer`, and the older JSON `locate` mode)
//   locate-cli   gates: the HairDeck yardstick, the memory bench, one-off queries
import PackageDescription

let package = Package(
    name: "mlx-locate-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Locate", targets: ["Locate"]),
        .library(name: "MLXLocate", targets: ["MLXLocate"]),
        .executable(name: "locate-cli", targets: ["locate-cli"]),
    ],
    dependencies: [
        // Qwen3-VL lives in MLXVLM. 3.33.0 is the first release on mlx-swift 0.33, which carries the
        // quantized-load leak fix (#501) the engine requires (AB-A-0249).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.33.0")),
        // The fleet fork, as mlx-engine-swift 0.82.1 takes it: 0.33.1 = upstream 0.33.0 with the
        // vendored MLX's M5 head-dim-split attention routes disabled (AB-D-0264). Same package
        // identity as ml-explore's.
        .package(url: "https://github.com/xocialize/mlx-swift", .upToNextMinor(from: "0.33.1")),
        // mlx-swift-lm 3.x leaves the tokenizer and downloader to the consumer.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // ≥0.81.0: contract 1.66.0, the imageLocate capability (AB-A-0295);
        // ≥0.82.1: mlx-swift from the fleet's 0.33.1 fork.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.82.1"),
    ],
    targets: [
        .target(
            name: "Locate",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]),
        .target(
            name: "MLXLocate",
            dependencies: [
                "Locate",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]),
        .executableTarget(
            name: "locate-cli",
            dependencies: [
                "Locate", "MLXLocate",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]),
        .testTarget(
            name: "LocateTests",
            dependencies: ["Locate"]),
        .testTarget(
            name: "MLXLocateTests",
            dependencies: [
                "MLXLocate",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]),
    ]
)
