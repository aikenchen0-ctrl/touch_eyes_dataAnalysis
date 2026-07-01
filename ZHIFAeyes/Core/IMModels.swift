import Foundation

struct IMUser: Hashable, Identifiable {
    let id: String
    var displayName: String
    var region: String
    var tags: [String]
}

struct IMConversation: Hashable, Identifiable {
    let id: String
    var title: String
    var participantIDs: [String]
    var lastMessageID: String?
    var lastActiveAt: Date
}

struct IMAccount: Hashable, Identifiable {
    let id: String
    var displayName: String
    var isPrimary: Bool
}

struct IMMessage: Hashable, Identifiable {
    let id: String
    let conversationID: String
    let senderID: String
    let createdAt: Date
    let isOutgoing: Bool
    let kind: IMMessageKind
    var replyToMessageID: String?
}

enum IMMessageKind: Hashable {
    case text(String)
    case image(URL)
    case audio(URL)
    case video(URL)
    case system(String)
    case file(URL, name: String)
    case card(String)
    case location(latitude: Double, longitude: Double)
    case link(URL)
    case custom(type: String, payload: Data)
}

struct IMRelationGraph {
    private(set) var edges: [String: Set<String>]

    init(edges: [String: Set<String>] = [:]) {
        self.edges = edges
    }

    mutating func addRelation(from sourceID: String, to targetID: String) {
        var set = edges[sourceID, default: []]
        set.insert(targetID)
        edges[sourceID] = set
    }

    func relatedIDs(for sourceID: String) -> [String] {
        Array(edges[sourceID] ?? [])
    }
}
