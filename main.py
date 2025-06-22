from tls_key_detector.detector import TLSKeyDetector

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "--dump":
        print("Usage: python main.py --dump memory.dmp")
        sys.exit(1)
    dump_path = sys.argv[2]
    detector = TLSKeyDetector(dump_path)
    results = detector.process_memory_dump()
    detector.export_results(results)
