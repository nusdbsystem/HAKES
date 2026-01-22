# Makefile for hakes-server
# Coordinates builds across embedder, gateway, searcher, and store components

.PHONY: all build clean test install help

# Default target
all: build

# Build all components
build:
# 	@echo "Building embedder/embed-worker..."
# 	$(MAKE) -C embedder/embed-worker
	@echo "Building gateway..."
	$(MAKE) -C gateway
	@echo "Building searcher..."
	$(MAKE) -C searcher
	@echo "Building store..."
	$(MAKE) -C store
	@echo "All components built successfully"

# Clean all components
clean:
# 	@echo "Cleaning embedder/embed-worker..."
# 	$(MAKE) -C embedder/embed-worker clean
	@echo "Cleaning gateway..."
	$(MAKE) -C gateway clean
	@echo "Cleaning searcher..."
	$(MAKE) -C searcher clean
	@echo "Cleaning store..."
	$(MAKE) -C store clean
	@echo "All components cleaned"

# Test all components
test:
# 	@echo "Testing embedder/embed-worker..."
# 	$(MAKE) -C embedder/embed-worker test
	@echo "Testing gateway..."
	$(MAKE) -C gateway test
	@echo "Testing searcher..."
	$(MAKE) -C searcher test
	@echo "Testing store..."
	$(MAKE) -C store test
	@echo "All tests completed"

# Install all components
install: build
# 	@echo "Installing embedder/embed-worker..."
# 	$(MAKE) -C embedder/embed-worker install
	@echo "Installing gateway..."
	$(MAKE) -C gateway install
	@echo "Installing searcher..."
	$(MAKE) -C searcher install
	@echo "Installing store..."
	$(MAKE) -C store install
	@echo "All components installed"

# Help target
help:
	@echo "Hakes Server Build System"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build all components (default)"
	@echo "  build     - Build all components"
	@echo "  clean     - Clean build artifacts from all components"
	@echo "  test      - Run tests for all components"
	@echo "  install   - Build and install all components"
	@echo "  help      - Display this help message"
	@echo ""
	@echo "Component structure:"
	@echo "  - embedder/embed-worker"
	@echo "  - gateway"
	@echo "  - searcher"
	@echo "  - store"
